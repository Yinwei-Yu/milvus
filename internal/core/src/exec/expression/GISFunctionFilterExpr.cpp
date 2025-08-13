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

#include "GISFunctionFilterExpr.h"
#include "common/EasyAssert.h"
#include "common/Geometry.h"
#include "common/Types.h"
#include "pb/plan.pb.h"
namespace milvus {
namespace exec {

#define GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(_DataType, method)          \
    auto execute_sub_batch = [](const _DataType* data,                         \
                                const bool* valid_data,                        \
                                const int32_t* offsets,                        \
                                const int size,                                \
                                TargetBitmapView res,                          \
                                TargetBitmapView valid_res,                    \
                                const Geometry& right_source) {                \
        for (int i = 0; i < size; ++i) {                                       \
            if (valid_data != nullptr && !valid_data[i]) {                     \
                res[i] = valid_res[i] = false;                                 \
                continue;                                                      \
            }                                                                  \
            res[i] =                                                           \
                Geometry(data[i].data(), data[i].size()).method(right_source); \
        }                                                                      \
    };                                                                         \
    int64_t processed_size = ProcessDataChunks<_DataType>(                     \
        execute_sub_batch, std::nullptr_t{}, res, valid_res, right_source);    \
    AssertInfo(processed_size == real_batch_size,                              \
               "internal error: expr processed rows {} not equal "             \
               "expect batch size {}",                                         \
               processed_size,                                                 \
               real_batch_size);                                               \
    return res_vec;

void
PhyGISFunctionFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    AssertInfo(expr_->column_.data_type_ == DataType::GEOMETRY,
               "unsupported data type: {}",
               expr_->column_.data_type_);
    if (is_index_mode_) {
        result = EvalForIndexSegment();
    } else {
        result = EvalForDataSegment();
    }
}

VectorPtr
PhyGISFunctionFilterExpr::EvalForDataSegment() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec = std::make_shared<ColumnVector>(
        TargetBitmap(real_batch_size), TargetBitmap(real_batch_size));
    TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
    TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);
    valid_res.set();

    auto& right_source = expr_->geometry_;

    // Choose underlying data type according to segment type to avoid element
    // size mismatch: Sealed segment variable column stores std::string_view;
    // Growing segment stores std::string.
    using SealedType = std::string_view;
    using GrowingType = std::string;

    switch (expr_->op_) {
        case proto::plan::GISFunctionFilterExpr_GISOp_Equals: {
            if (segment_->type() == SegmentType::Sealed) {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(SealedType, equals);
            } else {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(GrowingType, equals);
            }
        }
        case proto::plan::GISFunctionFilterExpr_GISOp_Touches: {
            if (segment_->type() == SegmentType::Sealed) {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(SealedType, touches);
            } else {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(GrowingType,
                                                           touches);
            }
        }
        case proto::plan::GISFunctionFilterExpr_GISOp_Overlaps: {
            if (segment_->type() == SegmentType::Sealed) {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(SealedType,
                                                           overlaps);
            } else {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(GrowingType,
                                                           overlaps);
            }
        }
        case proto::plan::GISFunctionFilterExpr_GISOp_Crosses: {
            if (segment_->type() == SegmentType::Sealed) {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(SealedType, crosses);
            } else {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(GrowingType,
                                                           crosses);
            }
        }
        case proto::plan::GISFunctionFilterExpr_GISOp_Contains: {
            if (segment_->type() == SegmentType::Sealed) {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(SealedType,
                                                           contains);
            } else {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(GrowingType,
                                                           contains);
            }
        }
        case proto::plan::GISFunctionFilterExpr_GISOp_Intersects: {
            if (segment_->type() == SegmentType::Sealed) {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(SealedType,
                                                           intersects);
            } else {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(GrowingType,
                                                           intersects);
            }
        }
        case proto::plan::GISFunctionFilterExpr_GISOp_Within: {
            if (segment_->type() == SegmentType::Sealed) {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(SealedType, within);
            } else {
                GEOMETRY_EXECUTE_SUB_BATCH_WITH_COMPARISON(GrowingType, within);
            }
        }
        default: {
            PanicInfo(NotImplemented,
                      "internal error: unknown GIS op : {}",
                      expr_->op_);
        }
    }
    return res_vec;
}

VectorPtr
PhyGISFunctionFilterExpr::EvalForIndexSegment() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    using Index = index::ScalarIndex<std::string>;

    // Prepare shared dataset for index query (coarse candidate set by R-Tree)
    auto ds = std::make_shared<milvus::Dataset>();
    ds->Set(milvus::index::OPERATOR_TYPE, expr_->op_);
    ds->Set(milvus::index::MATCH_VALUE, expr_->geometry_.to_wkb_string());

    /* ------------------------------------------------------------------
     * Prefetch: if coarse results are not cached yet, run a single R-Tree
     * query for all index chunks and cache their coarse bitmaps.
     * ------------------------------------------------------------------*/
    if (!coarse_cached_) {
        coarse_cache_.resize(num_index_chunk_);
        coarse_valid_cache_.resize(num_index_chunk_);

        for (size_t cid = 0; cid < num_index_chunk_; ++cid) {
            const Index& idx_ref =
                segment_->chunk_scalar_index<std::string>(field_id_, cid);
            auto* idx_ptr = const_cast<Index*>(&idx_ref);

            auto coarse = idx_ptr->Query(ds);
            coarse_cache_[cid] = std::move(coarse);

            auto valid = idx_ptr->IsNotNull();
            coarse_valid_cache_[cid] = std::move(valid);
        }
        coarse_cached_ = true;
    }

    TargetBitmap batch_result;
    TargetBitmap batch_valid;
    int processed_rows = 0;

    for (size_t i = current_index_chunk_; i < num_index_chunk_; ++i) {
        // 1) Build and cache refined bitmap for this chunk (coarse + exact)
        if (cached_index_chunk_id_ != static_cast<int64_t>(i)) {
            // Reuse segment-level coarse cache directly
            auto& coarse = coarse_cache_[i];
            auto& chunk_valid = coarse_valid_cache_[i];

            // Exact refinement
            TargetBitmap refined(coarse.size());
            const bool is_sealed = segment_->type() == SegmentType::Sealed;

            if (is_sealed) {
                auto [views, valid_vec] =
                    segment_->chunk_view<std::string_view>(field_id_, i);
                for (size_t pos = 0; pos < coarse.size(); ++pos) {
                    if (!coarse[pos])
                        continue;
                    if (!valid_vec.empty() && !valid_vec[pos])
                        continue;

                    const auto& wkb_view = views[pos];
                    Geometry left(wkb_view.data(), wkb_view.size(), false);
                    bool ok = false;
                    switch (expr_->op_) {
                        case proto::plan::GISFunctionFilterExpr_GISOp_Equals:
                            ok = left.equals(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Touches:
                            ok = left.touches(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Overlaps:
                            ok = left.overlaps(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Crosses:
                            ok = left.crosses(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Contains:
                            ok = left.contains(expr_->geometry_);
                            break;
                        case proto::plan::
                            GISFunctionFilterExpr_GISOp_Intersects:
                            ok = left.intersects(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Within:
                            ok = left.within(expr_->geometry_);
                            break;
                        default:
                            PanicInfo(NotImplemented,
                                      "unknown GIS op : {}",
                                      expr_->op_);
                    }
                    if (ok) {
                        refined.set(pos);
                    }
                }
            } else {  // Growing segment
                auto span = segment_->chunk_data<std::string>(field_id_, i);
                for (size_t pos = 0; pos < coarse.size(); ++pos) {
                    if (!coarse[pos])
                        continue;

                    const auto& wkb = span[pos];
                    Geometry left(wkb.data(), wkb.size(), false);
                    bool ok = false;
                    switch (expr_->op_) {
                        case proto::plan::GISFunctionFilterExpr_GISOp_Equals:
                            ok = left.equals(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Touches:
                            ok = left.touches(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Overlaps:
                            ok = left.overlaps(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Crosses:
                            ok = left.crosses(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Contains:
                            ok = left.contains(expr_->geometry_);
                            break;
                        case proto::plan::
                            GISFunctionFilterExpr_GISOp_Intersects:
                            ok = left.intersects(expr_->geometry_);
                            break;
                        case proto::plan::GISFunctionFilterExpr_GISOp_Within:
                            ok = left.within(expr_->geometry_);
                            break;
                        default:
                            PanicInfo(NotImplemented,
                                      "unknown GIS op : {}",
                                      expr_->op_);
                    }
                    if (ok) {
                        refined.set(pos);
                    }
                }
            }

            // Cache refined result for reuse by subsequent batches
            cached_index_chunk_id_ = i;
            cached_index_chunk_res_ = std::move(refined);
            // No need to copy valid bitmap into member; use coarse_valid_cache_[i] directly later
        }

        // 2) Append this chunk's cached results into current batch window
        const auto& chunk_valid_ref = coarse_valid_cache_[i];

        auto size = ProcessIndexOneChunk(batch_result,
                                         batch_valid,
                                         i,
                                         cached_index_chunk_res_,
                                         chunk_valid_ref,
                                         processed_rows);

        if (processed_rows + size >= batch_size_) {
            current_index_chunk_ = i;
            current_index_chunk_pos_ = i == current_index_chunk_
                                           ? current_index_chunk_pos_ + size
                                           : size;
            break;
        }
        processed_rows += size;
    }

    return std::make_shared<ColumnVector>(std::move(batch_result),
                                          std::move(batch_valid));
}

}  //namespace exec
}  // namespace milvus