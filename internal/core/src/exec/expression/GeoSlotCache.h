#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include <type_traits>

#include "common/Geometry.h"
#include "common/Types.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

// Per-chunk slot array storing decoded Geometry* pointers, one pointer per row within the chunk.
// Objects are published with CAS and read lock-free; freeing should be coordinated with segment lifecycle.
class GeoSlotChunk {
 public:
    explicit GeoSlotChunk(size_t size) : slots_(size) {
        for (auto& s : slots_) {
            s.store(nullptr, std::memory_order_relaxed);
        }
    }

    Geometry*
    Load(size_t idx) const {
        return slots_[idx].load(std::memory_order_acquire);
    }

    bool
    PublishIfEmpty(size_t idx, Geometry* g) {
        Geometry* expected = nullptr;
        return slots_[idx].compare_exchange_strong(
            expected, g, std::memory_order_release, std::memory_order_relaxed);
    }

    size_t
    Size() const {
        return slots_.size();
    }

    // Iterate and delete all published pointers; caller ensures no concurrent reads after this point.
    void
    ReleaseAll() {
        for (auto& s : slots_) {
            Geometry* p = s.load(std::memory_order_relaxed);
            if (p != nullptr) {
                delete p;
                s.store(nullptr, std::memory_order_relaxed);
            }
        }
    }

 private:
    std::vector<std::atomic<Geometry*>> slots_;
};

// Helper to hold per-type contiguous pointer ranges for binary search by data pointer.
template <typename T>
struct GeoChunkRange {
    const T* begin{nullptr};
    const T* end{nullptr};  // half-open [begin, end)
    int64_t chunk_id{-1};
};

template <typename T>
struct GeoRanges {
    std::vector<GeoChunkRange<T>> ranges;  // sorted by begin pointer
    std::atomic<bool> built{false};
};

// Per-segment per-field cache: lazily allocates GeoSlotChunk by chunk_id with size = segment_->chunk_size(field_id, chunk_id),
// and maintains per-type ranges for O(log C) location of a sub-batch base pointer.
class GeoSlotCache {
 public:
    GeoSlotChunk&
    GetOrCreateChunk(const segcore::SegmentInternalInterface* segment,
                     FieldId field_id,
                     int64_t chunk_id) {
        {
            std::lock_guard<std::mutex> g(mu_);
            auto it = chunks_.find(chunk_id);
            if (it != chunks_.end()) {
                return *(it->second);
            }
        }
        // Allocate outside lock then publish to minimize critical section time.
        size_t chunk_size =
            static_cast<size_t>(segment->chunk_size(field_id, chunk_id));
        auto up = std::make_unique<GeoSlotChunk>(chunk_size);
        std::lock_guard<std::mutex> g2(mu_);
        auto [it2, inserted] = chunks_.emplace(chunk_id, std::move(up));
        return *(it2->second);
    }

    void
    ReleaseAll() {
        std::lock_guard<std::mutex> g(mu_);
        for (auto& kv : chunks_) {
            kv.second->ReleaseAll();
        }
        chunks_.clear();
        sealed_.ranges.clear();
        sealed_.built.store(false, std::memory_order_relaxed);
        growing_.ranges.clear();
        growing_.built.store(false, std::memory_order_relaxed);
        sealed_ptr_index_.clear();
        sealed_ptr_index_built_.store(false, std::memory_order_relaxed);
    }

    // Build ranges once for a given element type (string_view for sealed, string for growing).
    template <typename T>
    void
    EnsureRangesBuilt(const segcore::SegmentInternalInterface* segment,
                      FieldId field_id) {
        auto& holder = Holder<T>();
        if (holder.built.load(std::memory_order_acquire))
            return;
        std::lock_guard<std::mutex> g(mu_);
        if (holder.built.load(std::memory_order_relaxed))
            return;
        int64_t nc = segment->num_chunk(field_id);
        holder.ranges.clear();
        holder.ranges.reserve(static_cast<size_t>(nc));
        for (int64_t cid = 0; cid < nc; ++cid) {
            int64_t sz = segment->chunk_size(field_id, cid);
            if (sz <= 0)
                continue;
            auto span = segment->chunk_data<T>(field_id, cid);
            GeoChunkRange<T> r;
            r.begin = span.data();
            r.end = span.data() + sz;
            r.chunk_id = cid;
            holder.ranges.emplace_back(r);
        }
        std::sort(holder.ranges.begin(),
                  holder.ranges.end(),
                  [](const GeoChunkRange<T>& a, const GeoChunkRange<T>& b) {
                      return a.begin < b.begin;
                  });
        holder.built.store(true, std::memory_order_release);
    }

    // Locate a data base pointer within built ranges; returns (range pointer, local_start).
    template <typename T>
    std::pair<const GeoChunkRange<T>*, size_t>
    LocateByData(const T* data) const {
        auto const& holder = ConstHolder<T>();
        if (!holder.built.load(std::memory_order_acquire))
            return {nullptr, 0};
        const auto& vec = holder.ranges;
        int64_t lo = 0, hi = static_cast<int64_t>(vec.size()) - 1;
        while (lo <= hi) {
            int64_t mid = (lo + hi) >> 1;
            const auto& r = vec[static_cast<size_t>(mid)];
            if (data < r.begin)
                hi = mid - 1;
            else if (data >= r.end)
                lo = mid + 1;
            else {
                size_t local_start = static_cast<size_t>(data - r.begin);
                return {&r, local_start};
            }
        }
        return {nullptr, 0};
    }

    // Build global pointer->(chunk_id, idx) index for sealed (string_view) once per (segment, field).
    void
    EnsureSealedPtrIndexBuilt(const segcore::SegmentInternalInterface* segment,
                              FieldId field_id) {
        if (sealed_ptr_index_built_.load(std::memory_order_acquire))
            return;
        std::lock_guard<std::mutex> g(mu_);
        if (sealed_ptr_index_built_.load(std::memory_order_relaxed))
            return;
        sealed_ptr_index_.clear();
        int64_t nc = segment->num_chunk(field_id);
        for (int64_t cid = 0; cid < nc; ++cid) {
            // Use chunk_view for variable-length sealed column; chunk_data/Span is not implemented for mmap variable columns.
            auto [views, valid] =
                segment->chunk_view<std::string_view>(field_id, cid);
            const bool* vptr = valid.data();
            for (size_t i = 0; i < views.size(); ++i) {
                if (vptr != nullptr && !vptr[i]) {
                    continue;
                }
                const void* key = static_cast<const void*>(views[i].data());
                sealed_ptr_index_.emplace(key, std::make_pair(cid, i));
            }
        }
        sealed_ptr_index_built_.store(true, std::memory_order_release);
    }

    // Lookup sealed pointer, returns (chunk_id, idx); found==false if not present.
    inline std::pair<bool, std::pair<int64_t, size_t>>
    LookupSealedPtr(const void* ptr) const {
        auto it = sealed_ptr_index_.find(ptr);
        if (it == sealed_ptr_index_.end())
            return {false, {0, 0}};
        return {true, it->second};
    }

 private:
    template <typename T>
    GeoRanges<T>&
    Holder();
    template <typename T>
    const GeoRanges<T>&
    ConstHolder() const;

 private:
    std::mutex mu_;
    std::unordered_map<int64_t, std::unique_ptr<GeoSlotChunk>> chunks_;

    GeoRanges<std::string_view> sealed_;
    GeoRanges<std::string> growing_;

    // Global pointer index for sealed: wkb_ptr -> (chunk_id, idx)
    std::unordered_map<const void*, std::pair<int64_t, size_t>>
        sealed_ptr_index_;
    std::atomic<bool> sealed_ptr_index_built_{false};
};

// Specializations to select holder by type.
template <>
inline GeoRanges<std::string_view>&
GeoSlotCache::Holder<std::string_view>() {
    return sealed_;
}

template <>
inline GeoRanges<std::string>&
GeoSlotCache::Holder<std::string>() {
    return growing_;
}

template <>
inline const GeoRanges<std::string_view>&
GeoSlotCache::ConstHolder<std::string_view>() const {
    return sealed_;
}

template <>
inline const GeoRanges<std::string>&
GeoSlotCache::ConstHolder<std::string>() const {
    return growing_;
}

// Manager mapping (segment_addr, field_id) to a GeoSlotCache instance.
class GeoSlotCacheManager {
 public:
    static GeoSlotCacheManager&
    Instance() {
        static GeoSlotCacheManager g;
        return g;
    }

    GeoSlotCache&
    GetOrCreate(const void* segment_addr, FieldId field_id) {
        std::lock_guard<std::mutex> g(mu_);
        auto key = std::make_pair(segment_addr, field_id.get());
        auto it = map_.find(key);
        if (it != map_.end())
            return *(it->second);
        auto up = std::make_unique<GeoSlotCache>();
        auto* raw = up.get();
        map_.emplace(key, std::move(up));
        return *raw;
    }

    void
    Remove(const void* segment_addr, FieldId field_id) {
        std::lock_guard<std::mutex> g(mu_);
        auto key = std::make_pair(segment_addr, field_id.get());
        map_.erase(key);
    }

 private:
    struct KeyHash {
        size_t
        operator()(const std::pair<const void*, int64_t>& k) const noexcept {
            auto h1 = std::hash<const void*>()(k.first);
            auto h2 = std::hash<int64_t>()(k.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    std::mutex mu_;
    std::unordered_map<std::pair<const void*, int64_t>,
                       std::unique_ptr<GeoSlotCache>,
                       KeyHash>
        map_;
};

}  // namespace exec
}  // namespace milvus