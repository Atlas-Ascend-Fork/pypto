/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_DUMP_H
#define PYPTO_DUMP_H

#if defined(__ENABLE_PTO_PRINTF__)
#ifndef __PTO_INCLUDE_INTERNAL_HEADERS__
#define __PTO_INCLUDE_INTERNAL_HEADERS__
#define __PYPTO_DUMP_DEFINED_INTERNAL_HEADERS__
#endif
#ifndef PYPTO_PTO_DEBUG
#define PYPTO_PTO_DEBUG 1
#define __PYPTO_DUMP_DEFINED_PTO_DEBUG__
#endif
#include <type_traits>
#include <utility>
#include "impl/utils/debug/pto_aicore_printf_impl.h"
#include "utils/debug/pto_dump.h"

template <typename T>
__aicore__ inline const __gm__ char* PyptoDumpDtypeName()
{
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, half>) {
        return "float16";
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return "bfloat16";
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return "int64";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return "uint64";
    } else if constexpr (std::is_same_v<T, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return "uint32";
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return "int16";
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return "uint16";
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return "int8";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return "uint8";
    } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
        return "float8_e4m3";
    } else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
        return "float8_e5m2";
    } else if constexpr (std::is_same_v<T, float8_e8m0_t>) {
        return "float8_e8m0";
    } else if constexpr (std::is_same_v<T, hifloat8_t>) {
        return "hifloat8";
    } else if constexpr (std::is_same_v<T, float4_e2m1x2_t>) {
        return "float4_e2m1";
    } else if constexpr (std::is_same_v<T, float4_e1m2x2_t>) {
        return "float4_e1m2";
    } else {
        return "unknown";
    }
}

template <pypto::aicore::DumpLayout Layout>
__aicore__ inline const __gm__ char* PyptoDumpLayoutName()
{
    if constexpr (Layout == pypto::aicore::DumpLayout::ND) {
        return "ND";
    } else if constexpr (Layout == pypto::aicore::DumpLayout::DN) {
        return "DN";
    } else if constexpr (Layout == pypto::aicore::DumpLayout::NZ) {
        return "NZ";
    } else if constexpr (Layout == pypto::aicore::DumpLayout::ZN) {
        return "ZN";
    } else if constexpr (Layout == pypto::aicore::DumpLayout::ZZ) {
        return "ZZ";
    } else {
        return "NN";
    }
}

template <typename T>
__aicore__ inline const __gm__ char* PyptoDumpMemoryName()
{
    if constexpr (T::Loc == pto::TileType::Mat) {
        return "L1 Buffer";
    } else if constexpr (T::Loc == pto::TileType::Acc) {
        return "L0C Buffer";
    } else if constexpr (T::Loc == pto::TileType::Bias) {
        return "BiasTable Buffer";
    } else if constexpr (T::Loc == pto::TileType::Scaling) {
        return "Fixpipe Buffer";
    } else if constexpr (T::Loc == pto::TileType::Vec) {
        return "UB";
    } else {
        return "unknown";
    }
}

template <pypto::aicore::DumpLayout Layout, typename T>
__aicore__ inline bool PyptoGetContiguousTensorDumpInfo(const pypto::aicore::DumpTensorView& view, uint32_t* dumpShape,
                                                        uint32_t& dumpSize)
{
    uint64_t logicalElements = 1U;
    for (uint32_t axis = 0U; axis < view.rank; ++axis) {
        const int64_t sourceExtent = view.sourceShape[axis];
        if (sourceExtent <= 0 || view.offsets[axis] != 0 || view.shape[axis] != sourceExtent) {
            return false;
        }

        const uint64_t extent = static_cast<uint64_t>(sourceExtent);
        if (logicalElements > 0xffffffffULL / extent) {
            return false;
        }
        logicalElements *= extent;
        dumpShape[axis] = static_cast<uint32_t>(sourceExtent);
    }

    if constexpr (Layout == pypto::aicore::DumpLayout::NZ) {
        if (view.sourceShape[view.rank - 2U] % view.innerRows != 0 ||
            view.sourceShape[view.rank - 1U] % view.innerCols != 0) {
            return false;
        }
    } else if constexpr (Layout == pypto::aicore::DumpLayout::ND) {
        uint64_t expectedStride = 1U;
        for (uint32_t axis = view.rank; axis > 0U; --axis) {
            const uint32_t index = axis - 1U;
            const uint64_t extent = static_cast<uint64_t>(view.sourceShape[index]);
            const int64_t stride = view.stride[index];
            if (stride <= 0 || (extent > 1U && static_cast<uint64_t>(stride) != expectedStride)) {
                return false;
            }
            expectedStride *= extent;
        }
    } else {
        return false;
    }

    if constexpr (std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>) {
        logicalElements = (logicalElements + 1U) / 2U;
    }
    dumpSize = static_cast<uint32_t>(logicalElements);
    return true;
}

template <uint32_t Rank>
__aicore__ inline const __gm__ char* PyptoDumpTensorMetadataFormat()
{
    static_assert(Rank >= 1U && Rank <= 5U, "dump tensor rank must be in the range [1, 5]");
    if constexpr (Rank == 1U) {
        return "[dump_tensor] Data Type: %s, Layout: %s, Source Shape: [%d], Offsets: [%d], Shape: [%d]\n";
    } else if constexpr (Rank == 2U) {
        return "[dump_tensor] Data Type: %s, Layout: %s, Source Shape: [%d, %d], Offsets: [%d, %d], "
               "Shape: [%d, %d]\n";
    } else if constexpr (Rank == 3U) {
        return "[dump_tensor] Data Type: %s, Layout: %s, Source Shape: [%d, %d, %d], Offsets: [%d, %d, %d], "
               "Shape: [%d, %d, %d]\n";
    } else if constexpr (Rank == 4U) {
        return "[dump_tensor] Data Type: %s, Layout: %s, Source Shape: [%d, %d, %d, %d], "
               "Offsets: [%d, %d, %d, %d], Shape: [%d, %d, %d, %d]\n";
    } else {
        return "[dump_tensor] Data Type: %s, Layout: %s, Source Shape: [%d, %d, %d, %d, %d], "
               "Offsets: [%d, %d, %d, %d, %d], Shape: [%d, %d, %d, %d, %d]\n";
    }
}

template <pypto::aicore::DumpLayout Layout, typename T, size_t... Indices>
__aicore__ inline void PyptoPrintTensorMetadata(const pypto::aicore::DumpTensorView& view,
                                                std::index_sequence<Indices...>)
{
    constexpr uint32_t rank = sizeof...(Indices);
    pypto::aicore::printf(PyptoDumpTensorMetadataFormat<rank>(), PyptoDumpDtypeName<T>(), PyptoDumpLayoutName<Layout>(),
                          static_cast<int>(view.sourceShape[Indices])..., static_cast<int>(view.offsets[Indices])...,
                          static_cast<int>(view.shape[Indices])...);
}

template <pypto::aicore::DumpLayout Layout, uint32_t Rank, typename T>
__aicore__ inline void PyptoDumpTensor(__gm__ T* src, uint32_t desc, const pypto::aicore::DumpTensorView& view)
{
    pipe_barrier(PIPE_ALL);
    PyptoPrintTensorMetadata<Layout, T>(view, std::make_index_sequence<Rank>{});
    uint32_t dumpShape[K_MAX_SHAPE_DIM];
    uint32_t dumpSize = 0U;
    if (PyptoGetContiguousTensorDumpInfo<Layout, T>(view, dumpShape, dumpSize)) {
        pypto::aicore::pto_dump_gm(src, desc, dumpSize, dumpShape, view.rank);
    } else {
        pypto::aicore::pto_dump_window<Layout>(src, desc, view);
    }
}

template <pypto::aicore::DumpLayout Layout, typename T>
__aicore__ inline void PyptoDumpTileWindow(T& src, uint32_t desc, int rowOffset, int colOffset, int requestedRows,
                                           int requestedCols)
{
    pipe_barrier(PIPE_ALL);

    const int runtimeRows = src.GetValidRow();
    const int runtimeCols = src.GetValidCol();
    const int availableRows = (runtimeRows < T::Rows ? runtimeRows : T::Rows) - rowOffset;
    const int availableCols = (runtimeCols < T::Cols ? runtimeCols : T::Cols) - colOffset;
    const int validRows = requestedRows > 0 && availableRows > 0 ?
                              (requestedRows < availableRows ? requestedRows : availableRows) :
                              0;
    const int validCols = requestedCols > 0 && availableCols > 0 ?
                              (requestedCols < availableCols ? requestedCols : availableCols) :
                              0;

    pypto::aicore::printf("[dump_tile] Data Type: %s, Layout: %s, Memory Space: %s, Source Shape: [%d, %d], "
                          "Offsets: [%d, %d], Shape: [%d, %d], Valid Shape: [%d, %d]\n",
                          PyptoDumpDtypeName<typename T::DType>(), PyptoDumpLayoutName<Layout>(),
                          PyptoDumpMemoryName<T>(), T::Rows, T::Cols, rowOffset, colOffset, requestedRows,
                          requestedCols, validRows, validCols);

    const int64_t sourceShape[2] = {T::Rows, T::Cols};
    const int64_t offsets[2] = {rowOffset, colOffset};
    const int64_t shape[2] = {validRows, validCols};
    constexpr int64_t innerRows = T::isBoxedLayout ? T::InnerRows : 1;
    constexpr int64_t innerCols = T::isBoxedLayout ? T::InnerCols : 1;
    const pypto::aicore::DumpTensorView view = {sourceShape, offsets, shape, nullptr, 2U, innerRows, innerCols};
    if constexpr (T::Loc == pto::TileType::Bias) {
        pypto::aicore::pto_dump_window<Layout>(reinterpret_cast<__biasbuf__ typename T::DType*>(src.data()), desc,
                                               view);
    } else {
        pypto::aicore::pto_dump_window<Layout>(src.data(), desc, view);
    }
}

#define PYPTO_DUMP_TENSOR(layout, rank, ...) PyptoDumpTensor<layout, rank>(__VA_ARGS__)
#define PYPTO_DUMP_TILE_WINDOW(layout, ...) PyptoDumpTileWindow<layout>(__VA_ARGS__)

#ifdef __PYPTO_DUMP_DEFINED_PTO_DEBUG__
#undef PYPTO_PTO_DEBUG
#undef __PYPTO_DUMP_DEFINED_PTO_DEBUG__
#endif
#ifdef __PYPTO_DUMP_DEFINED_INTERNAL_HEADERS__
#undef __PTO_INCLUDE_INTERNAL_HEADERS__
#undef __PYPTO_DUMP_DEFINED_INTERNAL_HEADERS__
#endif
#else
#define PYPTO_DUMP_TENSOR(...) ((void)0)
#define PYPTO_DUMP_TILE_WINDOW(...) ((void)0)
#endif

#endif // PYPTO_DUMP_H
