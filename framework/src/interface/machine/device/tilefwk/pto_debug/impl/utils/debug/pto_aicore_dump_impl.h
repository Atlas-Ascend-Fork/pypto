/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pto_aicore_dump_impl.h
 * \brief AICore tensor dump and window dump implementation.
 */
#ifndef IMPL_UTILS_DEBUG_PTO_AICORE_DUMP_IMPL_H
#define IMPL_UTILS_DEBUG_PTO_AICORE_DUMP_IMPL_H

#include <type_traits>

#ifndef ASCENDC_CPU_DEBUG
#include "impl/utils/sys_macros_impl.h"
#include "impl/utils/common_types.h"
#include "impl/utils/debug/pto_debug_types.h"
#include "impl/utils/debug/pto_debug_utils.h"

#if __NPU_ARCH__ == 2201
#include "impl/utils/debug/npu_arch_2201/pto_aicore_dump_utils.h"
#elif __NPU_ARCH__ == 3510
#include "impl/utils/debug/npu_arch_3510/pto_aicore_dump_utils.h"
#elif __NPU_ARCH__ == 9201
#include "impl/utils/debug/npu_arch_9201/pto_aicore_dump_utils.h"
#elif __NPU_ARCH__ == 5102
#include "impl/utils/debug/npu_arch_5102/pto_aicore_dump_utils.h"
#elif __NPU_ARCH__ == 5162
#include "impl/utils/debug/npu_arch_5162/pto_aicore_dump_utils.h"
#endif

namespace pypto::aicore {

template <AscendC::Hardware hardware, typename T, typename U>
__aicore__ inline void _set_dump_tlv_info(U src, __gm__ DumpTensorTlv* dumpTlv, uint32_t alignDumpLen, uint32_t desc,
                                          uint32_t dump_size)
{
    dumpTlv->type = static_cast<uint32_t>(DumpType::DUMP_TENSOR);
    dumpTlv->length = sizeof(DumpTensorTlv) - sizeof(uint32_t[2]) + alignDumpLen;
    dumpTlv->tensorAddr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src));
    dumpTlv->dataType = static_cast<uint32_t>(get_dump_datatype<T>());
    dumpTlv->desc = desc;
    dumpTlv->blockIdx = static_cast<uint32_t>(pto_debug_get_block_idx());
    dumpTlv->bufferId = static_cast<uint32_t>(0U);
    dumpTlv->position = static_cast<uint16_t>(hardware);
    dumpTlv->dim = static_cast<uint32_t>(0U);
    for (uint32_t i = 0; i < K_MAX_SHAPE_DIM; ++i) {
        dumpTlv->shape[i] = static_cast<uint32_t>(0U);
    }
    dumpTlv->resv1 = static_cast<uint32_t>(0U);
    dumpTlv->dumpSize = dump_size * sizeof(T);
    pto_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(dumpTlv));
}

template <typename Tlv>
__aicore__ inline void _set_dump_shape_info(__gm__ Tlv* dumpTlv, const uint32_t shapeDim, const uint32_t* shape)
{
    if (shapeDim <= 0 || shapeDim > K_MAX_SHAPE_DIM || shape == nullptr) {
        return;
    }
    dumpTlv->dim = static_cast<uint32_t>(shapeDim);
    for (uint32_t i = 0; i < K_MAX_SHAPE_DIM; ++i) {
        dumpTlv->shape[i] = i < shapeDim ? static_cast<uint32_t>(shape[i]) : static_cast<uint32_t>(1U);
    }
    pto_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(dumpTlv));
}

template <AscendC::Hardware hardware, typename T, typename U, typename Tlv>
__aicore__ inline uint32_t _set_dump_tlv_data(U src, __gm__ Tlv* dumpTlv, uint32_t alignDumpLen, uint32_t dump_size)
{
    __gm__ T* dumpDstAddr = reinterpret_cast<__gm__ T*>(dumpTlv + 1);

    if (dumpDstAddr == nullptr) {
        return 1;
    }
    if (hardware == AscendC::Hardware::GM && src == nullptr) {
        return 1;
    }

    sync_all();
    uint32_t ret = 0;
    uint32_t dumpLen = 0;
    if constexpr (hardware == AscendC::Hardware::GM) {
        dumpLen = dump_size * sizeof(T);
        ret = mem_copy_gm_to_gm(reinterpret_cast<__gm__ uint8_t*>(dumpDstAddr),
                                reinterpret_cast<__gm__ const uint8_t*>(src), dumpLen);
    } else if constexpr (hardware == AscendC::Hardware::UB) {
        dumpLen = alignDumpLen / PTO_ONE_DATABLOCK_SIZE;
        ret = mem_copy_ub_to_gm_impl(dumpDstAddr, src, static_cast<uint16_t>(dumpLen));
    } else if constexpr (hardware == AscendC::Hardware::L1) {
        ret = mem_copy_l1buf_to_gm_impl(dumpDstAddr, src, alignDumpLen);
    } else if constexpr (hardware == AscendC::Hardware::L0A) {
        ret = mem_copy_abuf_to_gm_impl(dumpDstAddr, src, alignDumpLen);
    } else if constexpr (hardware == AscendC::Hardware::L0B) {
        ret = mem_copy_bbuf_to_gm_impl(dumpDstAddr, src, alignDumpLen);
    } else if constexpr (hardware == AscendC::Hardware::L0C) {
        ret = mem_copy_cbuf_to_gm_impl(dumpDstAddr, src, alignDumpLen);
    } else if constexpr (hardware == AscendC::Hardware::BIAS) {
        ret = mem_copy_biasbuf_to_gm_impl(dumpDstAddr, src, alignDumpLen);
    } else if constexpr (hardware == AscendC::Hardware::FIXBUF) {
        ret = mem_copy_fbuf_to_gm_impl(dumpDstAddr, src, alignDumpLen);
    }
    sync_all();
    return ret;
}

template <AscendC::Hardware hardware>
__aicore__ constexpr inline bool _is_super_tensor_hardware_supported()
{
    return hardware == AscendC::Hardware::GM || hardware == AscendC::Hardware::UB ||
           hardware == AscendC::Hardware::L1 || hardware == AscendC::Hardware::L0C ||
           hardware == AscendC::Hardware::BIAS || hardware == AscendC::Hardware::FIXBUF;
}

template <AscendC::Hardware hardware>
__aicore__ constexpr inline uint32_t _get_super_tensor_payload_alignment()
{
    if constexpr (hardware == AscendC::Hardware::L0C) {
        return 16U * 16U * sizeof(uint32_t);
    }
    return PTO_ONE_DATABLOCK_SIZE;
}

template <AscendC::Hardware hardware>
__aicore__ inline uint32_t _get_super_tensor_payload_capacity(uint32_t ringBufLen, uint32_t headerLen)
{
    constexpr uint32_t payloadAlignment = _get_super_tensor_payload_alignment<hardware>();
    if (ringBufLen <= headerLen) {
        return 0U;
    }
    return ((ringBufLen - headerLen) / payloadAlignment) * payloadAlignment;
}

template <AscendC::Hardware hardware, typename T, typename U>
__aicore__ inline void _set_super_tensor_tlv_info(U src, __gm__ DumpSuperTensorTlv* dumpTlv, uint32_t alignDumpLen,
                                                  uint32_t desc, uint64_t tensorLength, uint32_t dumpSize,
                                                  const uint32_t* shape, uint32_t shapeDim)
{
    dumpTlv->type = static_cast<uint32_t>(DumpType::DUMP_SUPER_TENSOR);
    dumpTlv->length = sizeof(DumpSuperTensorTlv) - sizeof(uint32_t[2]) + alignDumpLen;
    dumpTlv->tensorAddr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src));
    dumpTlv->dataType = static_cast<uint32_t>(get_dump_datatype<T>());
    dumpTlv->desc = desc;
    dumpTlv->bufferId = 0U;
    dumpTlv->position = static_cast<uint16_t>(hardware);
    dumpTlv->blockIdx = static_cast<uint16_t>(pto_debug_get_block_idx());
    dumpTlv->dim = 0U;
    for (uint32_t i = 0; i < K_MAX_SHAPE_DIM; ++i) {
        dumpTlv->shape[i] = 0U;
    }
    dumpTlv->resv = 0U;
    dumpTlv->tensorLength = tensorLength;
    dumpTlv->tensorOffset = 0U;
    dumpTlv->dumpSize = dumpSize;
    _set_dump_shape_info(dumpTlv, shapeDim, shape);
    pto_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(dumpTlv));
}

__aicore__ inline void _set_super_tensor_body_tlv_info(__gm__ DumpSuperTensorBodyTlv* dumpTlv, uint32_t alignDumpLen,
                                                       uint64_t tensorLength, uint64_t tensorOffset, uint32_t dumpSize)
{
    dumpTlv->type = static_cast<uint32_t>(DumpType::DUMP_SUPER_TENSOR_BODY);
    dumpTlv->length = sizeof(DumpSuperTensorBodyTlv) - sizeof(uint32_t[2]) + alignDumpLen;
    dumpTlv->resv1 = 0U;
    dumpTlv->resv2 = 0U;
    dumpTlv->tensorLength = tensorLength;
    dumpTlv->tensorOffset = tensorOffset;
    dumpTlv->dumpSize = dumpSize;
    pto_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(dumpTlv));
}

template <AscendC::Hardware hardware, typename T, typename U>
__aicore__ inline void _pto_dump_super_tensor_impl(U src, uint32_t desc, uint64_t tensorLength, const uint32_t* shape,
                                                   uint32_t shapeDim, __gm__ DebugBlockHeadInfo* blockInfo)
{
    constexpr uint32_t payloadAlignment = _get_super_tensor_payload_alignment<hardware>();
    uint64_t tensorOffset = 0U;
    bool isFirst = true;
    while (tensorOffset < tensorLength) {
        const uint32_t headerLen = isFirst ? sizeof(DumpSuperTensorTlv) : sizeof(DumpSuperTensorBodyTlv);
        const uint32_t payloadCapacity = _get_super_tensor_payload_capacity<hardware>(blockInfo->ringBufLen, headerLen);
        if (payloadCapacity == 0U) {
            return;
        }
        const uint64_t remaining = tensorLength - tensorOffset;
        const uint32_t dumpSize = remaining > payloadCapacity ? payloadCapacity : static_cast<uint32_t>(remaining);
        const uint32_t alignDumpLen = align_up(dumpSize, payloadAlignment);
        const uint32_t tlvLen = headerLen + alignDumpLen;
        if (!check_ringbuf_space(blockInfo, tlvLen)) {
            return;
        }

        __gm__ uint8_t* tlvAddr = get_ringbuf_tlv_addr(blockInfo);
        U chunkSrc = src + tensorOffset / sizeof(T);
        if (isFirst) {
            auto* dumpTlv = reinterpret_cast<__gm__ DumpSuperTensorTlv*>(tlvAddr);
            _set_super_tensor_tlv_info<hardware, T>(src, dumpTlv, alignDumpLen, desc, tensorLength, dumpSize, shape,
                                                    shapeDim);
            if (_set_dump_tlv_data<hardware, T>(chunkSrc, dumpTlv, alignDumpLen, dumpSize / sizeof(T)) != 0U) {
                return;
            }
        } else {
            auto* dumpTlv = reinterpret_cast<__gm__ DumpSuperTensorBodyTlv*>(tlvAddr);
            _set_super_tensor_body_tlv_info(dumpTlv, alignDumpLen, tensorLength, tensorOffset, dumpSize);
            if (_set_dump_tlv_data<hardware, T>(chunkSrc, dumpTlv, alignDumpLen, dumpSize / sizeof(T)) != 0U) {
                return;
            }
        }

        __gm__ DebugBlockWriteInfo* writeInfo = get_block_write_info(blockInfo);
        update_write_info(writeInfo, tlvLen);
        tensorOffset += dumpSize;
        isFirst = false;
    }
}

template <AscendC::Hardware hardware, typename T, typename U>
__aicore__ inline void _pto_dump_impl(U src, uint32_t desc, uint32_t dump_size, const uint32_t* shape,
                                      const uint32_t shapeDim)
{
#if !(defined(ASCENDC_DUMP) && ASCENDC_DUMP == 0)
    __gm__ DebugBlockHeadInfo* blockInfo = get_block_info();
    if (dump_size <= 0 || blockInfo == nullptr) {
        return;
    }
    constexpr uint32_t dataBlockSize = 32U;
    const uint64_t tensorLength = static_cast<uint64_t>(dump_size) * sizeof(T);
    const uint64_t alignDumpLen64 = ((tensorLength + dataBlockSize - 1U) / dataBlockSize) * dataBlockSize;
    const uint64_t tlvLen64 = sizeof(DumpTensorTlv) + alignDumpLen64;
    if (tlvLen64 > blockInfo->ringBufLen) {
#if __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 9201
        if constexpr (_is_super_tensor_hardware_supported<hardware>()) {
            _pto_dump_super_tensor_impl<hardware, T>(src, desc, tensorLength, shape, shapeDim, blockInfo);
        }
#endif
        return;
    }
    const uint32_t alignDumpLen = static_cast<uint32_t>(alignDumpLen64);
    const uint32_t tlvLen = static_cast<uint32_t>(tlvLen64);
    if (!check_ringbuf_space(blockInfo, tlvLen)) {
        return;
    }
    __gm__ DumpTensorTlv* dumpTlv = reinterpret_cast<__gm__ DumpTensorTlv*>(get_ringbuf_tlv_addr(blockInfo));

    _set_dump_tlv_info<hardware, T>(src, dumpTlv, alignDumpLen, desc, dump_size);
    _set_dump_shape_info(dumpTlv, shapeDim, shape);
    if (_set_dump_tlv_data<hardware, T>(src, dumpTlv, alignDumpLen, dump_size) != 0) {
        return;
    }

    __gm__ DebugBlockWriteInfo* writeInfo = get_block_write_info(blockInfo);
    update_write_info(writeInfo, tlvLen);
#endif
}

__aicore__ inline void _pto_dump_shape_impl(const uint32_t shapeDim, const uint32_t* shape)
{
#if !(defined(ASCENDC_DUMP) && ASCENDC_DUMP == 0)
    __gm__ DebugBlockHeadInfo* blockInfo = get_block_info();
    if (blockInfo == nullptr) {
        return;
    }
    uint32_t tlvLen = sizeof(DumpShapeTlv);
    if (!check_ringbuf_space(blockInfo, tlvLen)) {
        return;
    }
    __gm__ DumpShapeTlv* shapeTlv = reinterpret_cast<__gm__ DumpShapeTlv*>(get_ringbuf_tlv_addr(blockInfo));
    shapeTlv->type = static_cast<uint32_t>(DumpType::DUMP_SHAPE);
    shapeTlv->length = tlvLen - sizeof(uint32_t[2]);
    shapeTlv->dim = shapeDim;
    for (uint32_t i = 0; i < 8; ++i) {
        shapeTlv->shape[i] = i < shapeDim ? shape[i] : 1;
    }
    shapeTlv->resv = static_cast<uint32_t>(0U);
    pto_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(shapeTlv));

    __gm__ DebugBlockWriteInfo* writeInfo = get_block_write_info(blockInfo);
    update_write_info(writeInfo, tlvLen);
#endif
}

template <AscendC::Hardware hardware, typename T, typename U>
__aicore__ inline void _pto_dump_local_impl(U input, uint32_t desc, uint32_t dump_size, const uint32_t* shape,
                                            const uint32_t shapeDim)
{
    uint64_t ctrlValue = get_ctrl();
    set_atomic_none();
    enable_pto_diagnostics();
    if (g_sysPrintFifoSpace != nullptr) {
        _pto_dump_impl<hardware, T>(input, desc, dump_size, shape, shapeDim);
    }
    set_ctrl(ctrlValue);
}

template <typename T>
struct DumpWindowRecord {
    __gm__ DebugBlockHeadInfo* blockInfo;
    __gm__ T* payload;
    uint32_t tlvLen;
};

template <AscendC::Hardware hardware, typename T, typename U>
__aicore__ inline bool _prepare_dump_window_record(DumpWindowRecord<T>& record, U src, uint32_t desc, uint32_t dumpSize,
                                                   const uint32_t* shape, uint32_t shapeDim)
{
#if defined(ASCENDC_DUMP) && ASCENDC_DUMP == 0
    return false;
#else
    record.blockInfo = get_block_info();
    if (dumpSize == 0U || record.blockInfo == nullptr) {
        return false;
    }

    constexpr uint64_t dataBlockSize = PTO_ONE_DATABLOCK_SIZE;
    const uint64_t dataBytes = static_cast<uint64_t>(dumpSize) * sizeof(T);
    const uint64_t alignedDataBytes = (dataBytes + dataBlockSize - 1U) / dataBlockSize * dataBlockSize;
    const uint64_t tlvBytes = sizeof(DumpTensorTlv) + alignedDataBytes;
    if (tlvBytes > record.blockInfo->ringBufLen || tlvBytes > 0xffffffffULL) {
        printf("[WARNING] dump_data desc=%u requires %llu bytes, but the per-core dump FIFO capacity is %u bytes; "
               "data was not dumped.\n",
               desc, static_cast<unsigned long long>(tlvBytes), record.blockInfo->ringBufLen);
        return false;
    }
    if (!check_ringbuf_space(record.blockInfo, static_cast<uint32_t>(tlvBytes))) {
        return false;
    }

    auto* dumpTlv = reinterpret_cast<__gm__ DumpTensorTlv*>(get_ringbuf_tlv_addr(record.blockInfo));
    record.payload = reinterpret_cast<__gm__ T*>(dumpTlv + 1);
    record.tlvLen = static_cast<uint32_t>(tlvBytes);
    _set_dump_tlv_info<hardware, T>(src, dumpTlv, static_cast<uint32_t>(alignedDataBytes), desc, dumpSize);
    _set_dump_shape_info(dumpTlv, shapeDim, shape);
    return true;
#endif
}

template <typename T>
__aicore__ inline void _publish_dump_window_record(const DumpWindowRecord<T>& record)
{
    sync_all();
    pto_entire_dcci(reinterpret_cast<__gm__ uint64_t*>(record.payload));
    update_write_info(get_block_write_info(record.blockInfo), record.tlvLen);
}

template <typename T>
constexpr __aicore__ inline bool _is_packed_fp4_dump_type()
{
    return std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
}

template <typename T>
using WindowDumpType = std::conditional_t<_is_packed_fp4_dump_type<T>(), float, T>;

template <typename T>
__aicore__ inline float _decode_fp4_dump_value(uint8_t value)
{
    constexpr float e2m1Values[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    constexpr float e1m2Values[8] = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F};
    const float magnitude = std::is_same_v<T, float4_e2m1x2_t> ? e2m1Values[value & 0x07U] : e1m2Values[value & 0x07U];
    return (value & 0x08U) == 0U ? magnitude : -magnitude;
}

template <DumpLayout layout>
__aicore__ inline int64_t _get_gm_matrix_offset(const DumpTensorView& view, int64_t batch, int64_t row, int64_t col)
{
    const uint32_t rowAxis = view.rank - 2U;
    const uint32_t colAxis = view.rank - 1U;
    if constexpr (layout == DumpLayout::NZ) {
        const int64_t paddedRows = view.stride[rowAxis];
        const int64_t c0 = view.stride[colAxis];
        return batch + (col / c0) * paddedRows * c0 + (row / 16) * 16 * c0 + (row % 16) * c0 + col % c0;
    }
    return batch + row * view.stride[rowAxis] + col * view.stride[colAxis];
}

__aicore__ inline int64_t _get_gm_batch_offset(const DumpTensorView& view, uint64_t batchIndex)
{
    int64_t batchOffset = 0;
    for (int32_t axis = static_cast<int32_t>(view.rank) - 3; axis >= 0; --axis) {
        const uint64_t extent = static_cast<uint64_t>(view.shape[axis]);
        const uint64_t coordinate = batchIndex % extent;
        batchIndex /= extent;
        batchOffset += (view.offsets[axis] + static_cast<int64_t>(coordinate)) * view.stride[axis];
    }
    return batchOffset;
}

template <typename T, typename SourcePtr>
__aicore__ inline void _copy_window_element(SourcePtr input, int64_t inputOffset, __gm__ WindowDumpType<T>* output,
                                            uint64_t outputOffset)
{
    if constexpr (_is_packed_fp4_dump_type<T>()) {
        const uint8_t inputByte = get_scalar_bitcode_value<T, uint8_t>(input[inputOffset / 2]);
        const uint8_t value = (inputByte >> ((inputOffset & 1) * 4)) & 0x0fU;
        output[outputOffset] = _decode_fp4_dump_value<T>(value);
    } else {
        output[outputOffset] = input[inputOffset];
    }
}

template <DumpLayout layout, typename T>
__aicore__ inline void _pack_gm_window(__gm__ T* input, const DumpTensorView& view, __gm__ WindowDumpType<T>* output)
{
    if (view.rank == 1U) {
        for (int64_t index = 0; index < view.shape[0]; ++index) {
            const int64_t inputOffset = (view.offsets[0] + index) * view.stride[0];
            _copy_window_element<T>(input, inputOffset, output, static_cast<uint64_t>(index));
        }
        return;
    }

    const uint32_t rowAxis = view.rank - 2U;
    const uint32_t colAxis = view.rank - 1U;
    uint64_t batchCount = 1U;
    for (uint32_t axis = 0U; axis < rowAxis; ++axis) {
        batchCount *= static_cast<uint64_t>(view.shape[axis]);
    }

    uint64_t outputOffset = 0U;
    for (uint64_t batchIndex = 0U; batchIndex < batchCount; ++batchIndex) {
        const int64_t batchOffset = _get_gm_batch_offset(view, batchIndex);
        if constexpr (layout == DumpLayout::ND) {
            for (int64_t row = 0; row < view.shape[rowAxis]; ++row) {
                int64_t inputOffset = batchOffset + (view.offsets[rowAxis] + row) * view.stride[rowAxis] +
                                      view.offsets[colAxis] * view.stride[colAxis];
                for (int64_t col = 0; col < view.shape[colAxis]; ++col) {
                    _copy_window_element<T>(input, inputOffset, output, outputOffset++);
                    inputOffset += view.stride[colAxis];
                }
            }
        } else {
            const int64_t c0 = view.stride[colAxis];
            const int64_t rowBegin = view.offsets[rowAxis];
            const int64_t rowEnd = rowBegin + view.shape[rowAxis];
            const int64_t colBegin = view.offsets[colAxis];
            const int64_t colEnd = colBegin + view.shape[colAxis];
            for (int64_t blockCol = colBegin / c0; blockCol < (colEnd + c0 - 1) / c0; ++blockCol) {
                const int64_t blockColBegin = blockCol * c0;
                const int64_t validColBegin = colBegin > blockColBegin ? colBegin : blockColBegin;
                const int64_t blockColEnd = blockColBegin + c0;
                const int64_t validColEnd = colEnd < blockColEnd ? colEnd : blockColEnd;
                for (int64_t blockRow = rowBegin / 16; blockRow < (rowEnd + 15) / 16; ++blockRow) {
                    const int64_t blockRowBegin = blockRow * 16;
                    const int64_t validRowBegin = rowBegin > blockRowBegin ? rowBegin : blockRowBegin;
                    const int64_t blockRowEnd = blockRowBegin + 16;
                    const int64_t validRowEnd = rowEnd < blockRowEnd ? rowEnd : blockRowEnd;
                    for (int64_t row = validRowBegin; row < validRowEnd; ++row) {
                        int64_t inputOffset = _get_gm_matrix_offset<layout>(view, batchOffset, row, validColBegin);
                        for (int64_t col = validColBegin; col < validColEnd; ++col) {
                            _copy_window_element<T>(input, inputOffset, output, outputOffset++);
                            ++inputOffset;
                        }
                    }
                }
            }
        }
    }
}

template <DumpLayout layout, typename T>
__aicore__ inline void _dump_tensor_window_impl(__gm__ T* input, uint32_t desc, const DumpTensorView& view)
{
    static_assert(layout == DumpLayout::ND || layout == DumpLayout::NZ, "unsupported GM dump layout");
    if (view.rank < 1U || view.rank > K_MAX_SHAPE_DIM || view.sourceShape == nullptr || view.offsets == nullptr ||
        view.shape == nullptr || view.stride == nullptr) {
        return;
    }

    uint32_t dumpShape[K_MAX_SHAPE_DIM];
    uint64_t outputElements = 1U;
    for (uint32_t i = 0U; i < view.rank; ++i) {
        if (view.sourceShape[i] <= 0 || view.offsets[i] < 0 || view.shape[i] <= 0 ||
            view.offsets[i] > view.sourceShape[i] - view.shape[i] ||
            outputElements > 0xffffffffULL / static_cast<uint64_t>(view.shape[i])) {
            return;
        }
        dumpShape[i] = static_cast<uint32_t>(view.shape[i]);
        outputElements *= static_cast<uint64_t>(dumpShape[i]);
    }

    const int64_t logicalOrigin = view.rank == 1U ? view.offsets[0] * view.stride[0] :
                                                    _get_gm_matrix_offset<layout>(view, _get_gm_batch_offset(view, 0U),
                                                                                  view.offsets[view.rank - 2U],
                                                                                  view.offsets[view.rank - 1U]);
    __gm__ T* windowInput = input + (_is_packed_fp4_dump_type<T>() ? logicalOrigin / 2 : logicalOrigin);
    using OutputType = WindowDumpType<T>;
    DumpWindowRecord<OutputType> record{};
    if (!_prepare_dump_window_record<AscendC::Hardware::GM>(
            record, windowInput, desc, static_cast<uint32_t>(outputElements), dumpShape, view.rank)) {
        return;
    }
    _pack_gm_window<layout>(input, view, record.payload);
    _publish_dump_window_record(record);
}

template <DumpLayout layout>
__aicore__ inline void _get_local_window_coordinate(const DumpTensorView& view, int64_t physicalOffset, int64_t& row,
                                                    int64_t& col)
{
    const int64_t rows = view.sourceShape[view.rank - 2U];
    const int64_t cols = view.sourceShape[view.rank - 1U];
    if constexpr (layout == DumpLayout::ND) {
        row = physicalOffset / cols;
        col = physicalOffset % cols;
    } else if constexpr (layout == DumpLayout::DN) {
        row = physicalOffset % rows;
        col = physicalOffset / rows;
    } else {
        const int64_t blockRows = rows / view.innerRows;
        const int64_t blockCols = cols / view.innerCols;
        const int64_t innerNumel = view.innerRows * view.innerCols;
        const int64_t block = physicalOffset / innerNumel;
        const int64_t inner = physicalOffset % innerNumel;
        if constexpr (layout == DumpLayout::NZ) {
            row = (block % blockRows) * view.innerRows + inner / view.innerCols;
            col = (block / blockRows) * view.innerCols + inner % view.innerCols;
        } else if constexpr (layout == DumpLayout::ZN) {
            row = (block / blockCols) * view.innerRows + inner % view.innerRows;
            col = (block % blockCols) * view.innerCols + inner / view.innerRows;
        } else if constexpr (layout == DumpLayout::ZZ) {
            row = (block / blockCols) * view.innerRows + inner / view.innerCols;
            col = (block % blockCols) * view.innerCols + inner % view.innerCols;
        } else {
            static_assert(layout == DumpLayout::NN, "unsupported local dump layout");
            row = (block % blockRows) * view.innerRows + inner % view.innerRows;
            col = (block / blockRows) * view.innerCols + inner / view.innerRows;
        }
    }
}

template <DumpLayout layout, typename T, typename SourcePtr>
__aicore__ inline void _pack_direct_window(SourcePtr input, const DumpTensorView& view,
                                           __gm__ WindowDumpType<T>* output)
{
    const int64_t rows = view.sourceShape[view.rank - 2U];
    const int64_t cols = view.sourceShape[view.rank - 1U];
    const int64_t rowBegin = view.offsets[view.rank - 2U];
    const int64_t rowEnd = rowBegin + view.shape[view.rank - 2U];
    const int64_t colBegin = view.offsets[view.rank - 1U];
    const int64_t colEnd = colBegin + view.shape[view.rank - 1U];
    uint64_t outputOffset = 0U;
    for (int64_t physicalOffset = 0; physicalOffset < rows * cols; ++physicalOffset) {
        int64_t row;
        int64_t col;
        _get_local_window_coordinate<layout>(view, physicalOffset, row, col);
        if (row >= rowBegin && row < rowEnd && col >= colBegin && col < colEnd) {
            _copy_window_element<T>(input, physicalOffset, output, outputOffset++);
        }
    }
}

struct DebugBusWindowChunk {
    uint32_t word0 = 0U;
    uint32_t word1 = 0U;
    uint32_t word2 = 0U;
    uint32_t word3 = 0U;
    uint32_t word4 = 0U;
    uint32_t word5 = 0U;
    uint32_t word6 = 0U;
    uint32_t word7 = 0U;

    __aicore__ inline uint32_t word(uint32_t index) const
    {
        switch (index) {
            case 0U:
                return word0;
            case 1U:
                return word1;
            case 2U:
                return word2;
            case 3U:
                return word3;
            case 4U:
                return word4;
            case 5U:
                return word5;
            case 6U:
                return word6;
            default:
                return word7;
        }
    }

    __aicore__ inline uint64_t bits(uint32_t bitOffset, uint32_t bitCount) const
    {
        uint64_t value = 0U;
        const uint32_t byteOffset = bitOffset >> 3;
        const uint32_t byteCount = (bitCount + 7U) >> 3;
        for (uint32_t i = 0U; i < byteCount; ++i) {
            const uint32_t byteIndex = byteOffset + i;
            const uint32_t byteValue = (word(byteIndex >> 2) >> ((byteIndex & 3U) << 3)) & 0xffU;
            value |= static_cast<uint64_t>(byteValue) << (i << 3);
        }
        value >>= bitOffset & 7U;
        return bitCount == 64U ? value : value & ((1ULL << bitCount) - 1U);
    }
};

template <AscendC::Hardware hardware, typename LocalPtr>
__aicore__ inline bool _read_local_window_chunk(LocalPtr src, DebugBusWindowChunk& chunk)
{
#if __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201
    const uint64_t debugBusAddr = get_debug_bus_addr_impl();
    if (debugBusAddr == 0U) {
        return false;
    }

    constexpr uint32_t readyMask = 0x80000000U;
    constexpr uint64_t timeoutCycles = 15U * 1000U * 1000U;
    constexpr uint64_t readBytes = get_debug_bus_loop_steplen<hardware>();
    constexpr uint32_t modelBase = get_debug_bus_model_base<hardware>();
    const uint64_t localAddr = reinterpret_cast<uint64_t>(src);
    uint32_t srcAddr;
    if constexpr (hardware == AscendC::Hardware::FIXBUF) {
        const uint64_t fbufType = localAddr >> PTO_DEBUG_BUS_FBUF_TYPE_SHIFT;
        srcAddr = (localAddr & PTO_DEBUG_BUS_FBUF_ADDR_MASK) >> get_debug_bus_local_addr_shift<hardware>();
        if (fbufType == 0U) {
            srcAddr += 0x2000U;
        } else if (fbufType == 1U) {
            srcAddr += 0x1000U;
        }
    } else {
        srcAddr = localAddr >> get_debug_bus_local_addr_shift<hardware>();
    }

    sync_all();
    debug_bus_write_reg(debugBusAddr, PTO_DEBUG_BUS_UNLOCK_OFFSET, PTO_DEBUG_BUS_UNLOCK_VALUE);
    debug_bus_write_reg(debugBusAddr, PTO_DEBUG_BUS_ADDR_LOW_OFFSET, srcAddr);
    debug_bus_write_reg(debugBusAddr, PTO_DEBUG_BUS_MODEL_BASE_OFFSET, modelBase);
    debug_bus_write_reg(debugBusAddr, PTO_DEBUG_BUS_CTRL_OFFSET, PTO_DEBUG_BUS_READ_MASK);

    const uint64_t start = pto_debug_get_system_cycle();
    while ((debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_BUSY_OFFSET) & readyMask) == 0U) {
        if (pto_debug_get_system_cycle() - start >= timeoutCycles) {
            return false;
        }
        ringbuf_wait_rts_sync<100U>();
    }

    ringbuf_wait_rts_sync<100U>();
    chunk.word0 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET);
    chunk.word1 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET + 4U);
    if constexpr (readBytes == 32U) {
        chunk.word2 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET + 8U);
        chunk.word3 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET + 12U);
        chunk.word4 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET + 16U);
        chunk.word5 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET + 20U);
        chunk.word6 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET + 24U);
        chunk.word7 = debug_bus_read_reg(debugBusAddr, PTO_DEBUG_BUS_DATA_OFFSET + 28U);
    }
    sync_all();
    return true;
#else
    (void)src;
    (void)chunk;
    return false;
#endif
}

template <AscendC::Hardware hardware, DumpLayout layout, typename T, typename LocalPtr>
__aicore__ inline bool _pack_debug_bus_window(LocalPtr input, const DumpTensorView& view,
                                              __gm__ WindowDumpType<T>* output)
{
#if __NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201
    constexpr uint32_t transferBytes = get_debug_bus_loop_steplen<hardware>();
    constexpr uint32_t packedElements = _is_packed_fp4_dump_type<T>() ? 2U : 1U;
    constexpr uint32_t logicalElementsPerChunk = transferBytes * packedElements / sizeof(T);
    static_assert(logicalElementsPerChunk <= 64U, "local dump window mask only supports up to 64 elements per chunk");

    const int64_t rows = view.sourceShape[view.rank - 2U];
    const int64_t cols = view.sourceShape[view.rank - 1U];
    const int64_t physicalNumel = rows * cols;
    const int64_t storageBytes = (physicalNumel + packedElements - 1U) / packedElements * sizeof(T);
    if (storageBytes % transferBytes != 0) {
        return false;
    }

    const int64_t rowBegin = view.offsets[view.rank - 2U];
    const int64_t rowEnd = rowBegin + view.shape[view.rank - 2U];
    const int64_t colBegin = view.offsets[view.rank - 1U];
    const int64_t colEnd = colBegin + view.shape[view.rank - 1U];
    uint64_t outputOffset = 0U;
    DebugBusWindowChunk chunk{};
    for (int64_t physicalBase = 0; physicalBase < physicalNumel; physicalBase += logicalElementsPerChunk) {
        uint64_t windowMask = 0U;
        for (uint32_t inner = 0U; inner < logicalElementsPerChunk && physicalBase + inner < physicalNumel; ++inner) {
            int64_t row;
            int64_t col;
            _get_local_window_coordinate<layout>(view, physicalBase + inner, row, col);
            if (row >= rowBegin && row < rowEnd && col >= colBegin && col < colEnd) {
                windowMask |= 1ULL << inner;
            }
        }
        if (windowMask == 0U) {
            continue;
        }

        auto localChunk = input + physicalBase / packedElements;
        if (!_read_local_window_chunk<hardware>(localChunk, chunk)) {
            return false;
        }
        constexpr uint32_t elementBits = _is_packed_fp4_dump_type<T>() ? 4U : sizeof(T) * 8U;
        for (uint32_t inner = 0U; inner < logicalElementsPerChunk && physicalBase + inner < physicalNumel; ++inner) {
            if ((windowMask & (1ULL << inner)) != 0U) {
                const uint64_t value = chunk.bits(inner * elementBits, elementBits);
                if constexpr (_is_packed_fp4_dump_type<T>()) {
                    output[outputOffset++] = _decode_fp4_dump_value<T>(static_cast<uint8_t>(value));
                } else {
                    output[outputOffset++] = get_scalar_bitcode_value<uint64_t, T>(value);
                }
            }
        }
    }
    return true;
#else
    (void)input;
    (void)view;
    (void)output;
    return false;
#endif
}

template <AscendC::Hardware hardware, DumpLayout layout, typename T, typename SourcePtr>
__aicore__ inline void _dump_tile_window_impl(SourcePtr input, uint32_t desc, const DumpTensorView& view)
{
    static_assert(layout == DumpLayout::ND || layout == DumpLayout::DN || layout == DumpLayout::NZ ||
                      layout == DumpLayout::ZN || layout == DumpLayout::ZZ || layout == DumpLayout::NN,
                  "unsupported local dump layout");
    if (view.rank != 2U || view.sourceShape == nullptr || view.offsets == nullptr || view.shape == nullptr ||
        view.innerRows <= 0 || view.innerCols <= 0 || view.sourceShape[0] <= 0 || view.sourceShape[1] <= 0 ||
        view.sourceShape[0] % view.innerRows != 0 || view.sourceShape[1] % view.innerCols != 0 || view.offsets[0] < 0 ||
        view.offsets[1] < 0 || view.shape[0] <= 0 || view.shape[1] <= 0 ||
        view.offsets[0] > view.sourceShape[0] - view.shape[0] ||
        view.offsets[1] > view.sourceShape[1] - view.shape[1]) {
        return;
    }
    const uint64_t outputElements64 = static_cast<uint64_t>(view.shape[0]) * static_cast<uint64_t>(view.shape[1]);
    if (outputElements64 > 0xffffffffULL) {
        return;
    }

    using OutputType = WindowDumpType<T>;
    const uint32_t outputElements = static_cast<uint32_t>(outputElements64);
    uint32_t dumpShape[2] = {static_cast<uint32_t>(view.shape[0]), static_cast<uint32_t>(view.shape[1])};
    DumpWindowRecord<OutputType> record{};
    if (!_prepare_dump_window_record<hardware>(record, input, desc, outputElements, dumpShape, 2U)) {
        return;
    }

    if constexpr (hardware == AscendC::Hardware::UB) {
        _pack_direct_window<layout, T>(input, view, record.payload);
    } else {
        if (!_pack_debug_bus_window<hardware, layout, T>(input, view, record.payload)) {
            return;
        }
    }
    _publish_dump_window_record(record);
}
template <AscendC::Hardware hardware, DumpLayout layout, typename T, typename U>
__aicore__ inline void _pto_dump_window_impl(U input, uint32_t desc, const DumpTensorView& view)
{
    uint64_t ctrlValue = get_ctrl();
    set_atomic_none();
    enable_pto_diagnostics();
    if (g_sysPrintFifoSpace != nullptr) {
        if constexpr (hardware == AscendC::Hardware::GM) {
            _dump_tensor_window_impl<layout>(input, desc, view);
        } else {
            _dump_tile_window_impl<hardware, layout, T>(input, desc, view);
        }
    }
    set_ctrl(ctrlValue);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dump_size,
                                                             const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::GM, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dump_size,
                                                               const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::UB, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_abuf(__ca__ T* input, uint32_t desc, uint32_t dump_size,
                                                               const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::L0A, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_bbuf(__cb__ T* input, uint32_t desc, uint32_t dump_size,
                                                               const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::L0B, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_cbuf(__cc__ T* input, uint32_t desc, uint32_t dump_size,
                                                               const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::L0C, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_l1buf(__cbuf__ T* input, uint32_t desc, uint32_t dump_size,
                                                                const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::L1, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__biasbuf__ T* input, uint32_t desc, uint32_t dump_size,
                                                          const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::BIAS, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__fbuf__ T* input, uint32_t desc, uint32_t dump_size,
                                                          const uint32_t* shape, const uint32_t shapeDim)
{
    _pto_dump_local_impl<AscendC::Hardware::FIXBUF, T>(input, desc, dump_size, shape, shapeDim);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dump_size)
{
    _pto_dump_local_impl<AscendC::Hardware::GM, T>(input, desc, dump_size, nullptr, 0);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    _pto_dump_local_impl<AscendC::Hardware::UB, T>(input, desc, dump_size, nullptr, 0);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_abuf(__ca__ T* input, uint32_t desc, uint32_t dump_size)
{
    _pto_dump_local_impl<AscendC::Hardware::L0A, T>(input, desc, dump_size, nullptr, 0);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_bbuf(__cb__ T* input, uint32_t desc, uint32_t dump_size)
{
    _pto_dump_local_impl<AscendC::Hardware::L0B, T>(input, desc, dump_size, nullptr, 0);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_cbuf(__cc__ T* input, uint32_t desc, uint32_t dump_size)
{
    _pto_dump_local_impl<AscendC::Hardware::L0C, T>(input, desc, dump_size, nullptr, 0);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_l1buf(__cbuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    _pto_dump_local_impl<AscendC::Hardware::L1, T>(input, desc, dump_size, nullptr, 0);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__gm__ T* input, uint32_t desc, uint32_t dump_size)
{
    pto_dump_gm(input, desc, dump_size);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__ubuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    pto_dump_ubuf(input, desc, dump_size);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__cc__ T* input, uint32_t desc, uint32_t dump_size)
{
    pto_dump_cbuf(input, desc, dump_size);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__cbuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    pto_dump_l1buf(input, desc, dump_size);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__biasbuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    pto_dump(input, desc, dump_size, nullptr, 0);
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(__fbuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    pto_dump(input, desc, dump_size, nullptr, 0);
}

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__gm__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view)
{
    _pto_dump_window_impl<AscendC::Hardware::GM, layout, T>(input, desc, view);
}

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__ubuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view)
{
    _pto_dump_window_impl<AscendC::Hardware::UB, layout, T>(input, desc, view);
}

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__cc__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view)
{
    _pto_dump_window_impl<AscendC::Hardware::L0C, layout, T>(input, desc, view);
}

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__cbuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view)
{
    _pto_dump_window_impl<AscendC::Hardware::L1, layout, T>(input, desc, view);
}

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__biasbuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view)
{
    _pto_dump_window_impl<AscendC::Hardware::BIAS, layout, T>(input, desc, view);
}

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(__fbuf__ T* input, uint32_t desc,
                                                                 const DumpTensorView& view)
{
    _pto_dump_window_impl<AscendC::Hardware::FIXBUF, layout, T>(input, desc, view);
}

} // namespace pypto::aicore
#else
#include <cstdio>

namespace pypto::aicore {
template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dump_size)
{
    assert(false && "pto_dump_gm is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    assert(false && "pto_dump_ubuf is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_cbuf(__cc__ T* input, uint32_t desc, uint32_t dump_size)
{
    assert(false && "pto_dump_cbuf is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_l1buf(__cbuf__ T* input, uint32_t desc, uint32_t dump_size)
{
    assert(false && "pto_dump_l1buf is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_gm(__gm__ T* input, uint32_t desc, uint32_t dump_size,
                                                             const uint32_t* shape, const uint32_t shapeDim)
{
    assert(false && "pto_dump_gm is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_ubuf(__ubuf__ T* input, uint32_t desc, uint32_t dump_size,
                                                               const uint32_t* shape, const uint32_t shapeDim)
{
    assert(false && "pto_dump_ubuf is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_cbuf(__cc__ T* input, uint32_t desc, uint32_t dump_size,
                                                               const uint32_t* shape, const uint32_t shapeDim)
{
    assert(false && "pto_dump_cbuf is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_l1buf(__cbuf__ T* input, uint32_t desc, uint32_t dump_size,
                                                                const uint32_t* shape, const uint32_t shapeDim)
{
    assert(false && "pto_dump_l1buf is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(T* input, uint32_t desc, uint32_t dump_size)
{
    assert(false && "pto_dump is not supported in cpu mode.");
}

template <typename T>
__aicore__ static __attribute__((noinline)) void pto_dump(T* input, uint32_t desc, uint32_t dump_size,
                                                          const uint32_t* shape, const uint32_t shapeDim)
{
    assert(false && "pto_dump is not supported in cpu mode.");
}

__aicore__ inline void _pto_dump_shape_impl(const uint32_t shapeDim, const uint32_t* shape)
{
    assert(false && "_pto_dump_shape_impl is not supported in cpu mode.");
}

template <DumpLayout layout, typename T>
__aicore__ static __attribute__((noinline)) void pto_dump_window(T* input, uint32_t desc, const DumpTensorView& view)
{
    assert(false && "pto_dump_window is not supported in cpu mode.");
}
} // namespace pypto::aicore

using namespace pypto::aicore;
#endif

#if defined(__UNDEF_PTO_INCLUDE_INTERNAL_HEADERS_PTO_AICORE_DUMP_IMPL__)
#undef __PTO_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_PTO_INCLUDE_INTERNAL_HEADERS_PTO_AICORE_DUMP_IMPL__
#endif

#endif // IMPL_UTILS_DEBUG_PTO_AICORE_DUMP_IMPL_H
