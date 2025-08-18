// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "RTreeIndexWrapper.h"
#include "common/EasyAssert.h"
#include "log/Log.h"
#include "pb/plan.pb.h"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <mutex>
#include <nlohmann/json.hpp>
#include "common/FieldDataInterface.h"

namespace milvus::index {

namespace bgi = boost::geometry::index;

RTreeIndexWrapper::RTreeIndexWrapper(std::string& path, bool is_build_mode)
    : index_path_(path), is_build_mode_(is_build_mode) {
    if (is_build_mode_) {
        std::filesystem::path dir_path =
            std::filesystem::path(path).parent_path();
        if (!dir_path.empty()) {
            std::filesystem::create_directories(dir_path);
        }
        // Start with an empty rtree for dynamic insertions
        rtree_ = std::make_unique<RTree>();
    }
}

RTreeIndexWrapper::~RTreeIndexWrapper() = default;

void
RTreeIndexWrapper::add_geometry(const uint8_t* wkb_data,
                                size_t len,
                                int64_t row_offset) {
    AssertInfo(is_build_mode_, "Cannot add geometry in load mode");
    std::unique_lock<std::shared_mutex> guard(rtree_mutex_);
    if (!rtree_) {
        rtree_ = std::make_unique<RTree>();
    }

    // Parse WKB data to OGR geometry
    OGRGeometry* geom = nullptr;
    OGRErr err =
        OGRGeometryFactory::createFromWkb(wkb_data, nullptr, &geom, len);

    if (err != OGRERR_NONE || geom == nullptr) {
        LOG_ERROR("Failed to parse WKB data for row {}", row_offset);
        return;
    }

    // Get bounding box
    double minX, minY, maxX, maxY;
    get_bounding_box(geom, minX, minY, maxX, maxY);

    // Create Boost box and insert
    Box box(Point(minX, minY), Point(maxX, maxY));
    Value val(box, row_offset);
    values_.push_back(val);
    rtree_->insert(val);

    // Clean up
    OGRGeometryFactory::destroyGeometry(geom);
}

// No IDataStream; bulk-load implemented directly for Boost R-tree

void
RTreeIndexWrapper::bulk_load_from_field_data(
    const std::vector<std::shared_ptr<::milvus::FieldDataBase>>& field_datas,
    bool nullable) {
    AssertInfo(is_build_mode_, "Cannot bulk load in load mode");
    std::unique_lock<std::shared_mutex> guard(rtree_mutex_);
    std::vector<Value> local_values;
    local_values.reserve(1024);
    int64_t absolute_offset = 0;
    for (const auto& fd : field_datas) {
        const auto n = fd->get_num_rows();
        for (int64_t i = 0; i < n; ++i, ++absolute_offset) {
            const bool is_nullable_effective = nullable || fd->IsNullable();
            if (is_nullable_effective && !fd->is_valid(i)) {
                continue;
            }
            const auto* wkb_str =
                static_cast<const std::string*>(fd->RawValue(i));
            if (wkb_str == nullptr || wkb_str->empty()) {
                continue;
            }
            OGRGeometry* geom = nullptr;
            auto err = OGRGeometryFactory::createFromWkb(
                reinterpret_cast<const uint8_t*>(wkb_str->data()),
                nullptr,
                &geom,
                wkb_str->size());
            if (err != OGRERR_NONE || geom == nullptr) {
                continue;
            }
            OGREnvelope env;
            geom->getEnvelope(&env);
            OGRGeometryFactory::destroyGeometry(geom);
            Box box(Point(env.MinX, env.MinY), Point(env.MaxX, env.MaxY));
            local_values.emplace_back(box, absolute_offset);
        }
    }
    values_.swap(local_values);
    rtree_ = std::make_unique<RTree>(values_.begin(), values_.end());
    LOG_INFO("R-Tree bulk load (Boost) completed with {} entries",
             values_.size());
}

void
RTreeIndexWrapper::finish() {
    // Guard against repeated invocations which could otherwise attempt to
    // release resources multiple times (e.g. BuildWithRawDataForUT() calls
    // finish(), and Upload() may call it again).
    std::unique_lock<std::shared_mutex> guard(rtree_mutex_);
    if (finished_) {
        LOG_DEBUG("RTreeIndexWrapper::finish() called more than once, skip.");
        return;
    }

    AssertInfo(is_build_mode_, "Cannot finish in load mode");

    // Persist to disk: write meta and binary data file
    try {
        // Write binary rtree data
        std::ofstream data_out(index_path_ + ".bgi",
                               std::ios::binary | std::ios::trunc);
        if (!data_out.good()) {
            PanicInfo(
                ErrorCode::UnexpectedError,
                fmt::format("Failed to open {}.bgi for writing", index_path_));
        }
        const char magic[6] = {'B', 'G', 'R', 'T', 'R', '1'};
        data_out.write(magic, sizeof(magic));
        uint64_t count = static_cast<uint64_t>(values_.size());
        data_out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& v : values_) {
            const auto& b = v.first;
            double minx = b.min_corner().get<0>();
            double miny = b.min_corner().get<1>();
            double maxx = b.max_corner().get<0>();
            double maxy = b.max_corner().get<1>();
            data_out.write(reinterpret_cast<const char*>(&minx),
                           sizeof(double));
            data_out.write(reinterpret_cast<const char*>(&miny),
                           sizeof(double));
            data_out.write(reinterpret_cast<const char*>(&maxx),
                           sizeof(double));
            data_out.write(reinterpret_cast<const char*>(&maxy),
                           sizeof(double));
            int64_t id = v.second;
            data_out.write(reinterpret_cast<const char*>(&id), sizeof(int64_t));
        }
        data_out.close();

        // Write meta json
        nlohmann::json meta;
        // index/leaf capacities are not used in Boost implementation
        meta["dimension"] = dimension_;
        meta["bgi_file"] = std::string(index_path_ + ".bgi");
        meta["count"] = static_cast<uint64_t>(values_.size());

        std::ofstream ofs(index_path_ + ".meta.json", std::ios::trunc);
        ofs << meta.dump();
        ofs.close();
        LOG_INFO("R-Tree meta written: {}.meta.json", index_path_);
    } catch (const std::exception& e) {
        LOG_WARN("Failed to write R-Tree files: {}", e.what());
    }

    finished_ = true;

    LOG_INFO("R-Tree index (Boost) finished building and saved to {}",
             index_path_);
}

void
RTreeIndexWrapper::load() {
    AssertInfo(!is_build_mode_, "Cannot load in build mode");

    std::unique_lock<std::shared_mutex> guard(rtree_mutex_);
    try {
        // Read meta (optional)
        try {
            std::ifstream ifs(index_path_ + ".meta.json");
            if (ifs.good()) {
                auto meta = nlohmann::json::parse(ifs);
                // index/leaf capacities are ignored for Boost implementation
                if (meta.contains("dimension"))
                    dimension_ = meta["dimension"].get<uint32_t>();
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to read meta json: {}", e.what());
        }

        // Read binary data
        std::ifstream data_in(index_path_ + ".bgi", std::ios::binary);
        if (!data_in.good()) {
            PanicInfo(
                ErrorCode::UnexpectedError,
                fmt::format("Failed to open {}.bgi for reading", index_path_));
        }
        char magic[6];
        data_in.read(magic, sizeof(magic));
        if (std::string(magic, sizeof(magic)) != std::string("BGRTR1", 6)) {
            PanicInfo(ErrorCode::UnexpectedError,
                      "Invalid R-Tree binary magic");
        }
        uint64_t count = 0;
        data_in.read(reinterpret_cast<char*>(&count), sizeof(count));
        values_.clear();
        values_.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            double minx, miny, maxx, maxy;
            int64_t id;
            data_in.read(reinterpret_cast<char*>(&minx), sizeof(double));
            data_in.read(reinterpret_cast<char*>(&miny), sizeof(double));
            data_in.read(reinterpret_cast<char*>(&maxx), sizeof(double));
            data_in.read(reinterpret_cast<char*>(&maxy), sizeof(double));
            data_in.read(reinterpret_cast<char*>(&id), sizeof(int64_t));
            Box box(Point(minx, miny), Point(maxx, maxy));
            values_.emplace_back(box, id);
        }
        data_in.close();

        rtree_ = std::make_unique<RTree>(values_.begin(), values_.end());

        LOG_INFO("R-Tree index (Boost) loaded from {}", index_path_);
    } catch (const std::exception& e) {
        PanicInfo(ErrorCode::UnexpectedError,
                  fmt::format("Failed to load R-Tree index from {}: {}",
                              index_path_,
                              e.what()));
    }
}

void
RTreeIndexWrapper::query_candidates(proto::plan::GISFunctionFilterExpr_GISOp op,
                                    const OGRGeometry& query_geom,
                                    std::vector<int64_t>& candidate_offsets) {
    AssertInfo(rtree_ != nullptr, "R-Tree index not initialized");

    candidate_offsets.clear();

    // Get bounding box of query geometry
    double minX, minY, maxX, maxY;
    get_bounding_box(&query_geom, minX, minY, maxX, maxY);

    // Create query box
    Box query_box(Point(minX, minY), Point(maxX, maxY));

    // Perform coarse intersection query
    auto t0 = std::chrono::high_resolution_clock::now();
    LOG_INFO("R-Tree query start");
    std::vector<Value> results;
    {
        std::shared_lock<std::shared_mutex> guard(rtree_mutex_);
        rtree_->query(bgi::intersects(query_box), std::back_inserter(results));
    }
    candidate_offsets.reserve(results.size());
    for (const auto& v : results) {
        candidate_offsets.push_back(v.second);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    LOG_INFO("R-Tree query done, candidates = {}, cost = {} us",
             candidate_offsets.size(),
             elapsed_us);

    LOG_DEBUG("R-Tree query returned {} candidates for operation {}",
              candidate_offsets.size(),
              static_cast<int>(op));
}

void
RTreeIndexWrapper::get_bounding_box(const OGRGeometry* geom,
                                    double& minX,
                                    double& minY,
                                    double& maxX,
                                    double& maxY) {
    AssertInfo(geom != nullptr, "Geometry is null");

    OGREnvelope env;
    geom->getEnvelope(&env);

    minX = env.MinX;
    minY = env.MinY;
    maxX = env.MaxX;
    maxY = env.MaxY;
}

int64_t
RTreeIndexWrapper::count() const {
    if (!rtree_) {
        return 0;
    }
    return static_cast<int64_t>(rtree_->size());
}

// index/leaf capacity setters removed; not applicable for Boost rtree
}  // namespace milvus::index