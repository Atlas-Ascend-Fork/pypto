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
 * \\file range.cpp
 * \\brief
 */

#include <climits>
#include <cmath>
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

static bool SupportsInt64Range()
{
    static const std::unordered_map<NPUArch, bool> supportsInt64 = {
        {NPUArch::DAV_1001, false}, {NPUArch::DAV_2201, false}, {NPUArch::DAV_3510, true},  {NPUArch::DAV_9201, true},
        {NPUArch::DAV_3003, false}, {NPUArch::DAV_3113, false}, {NPUArch::DAV_5101, false},
    };
    auto arch = Platform::Instance().GetSoc().GetNPUArch();
    auto it = supportsInt64.find(arch);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, it != supportsInt64.end())
        << "Range int64 support is not configured for architecture: " << NPUArchToString(arch);
    return it->second;
}

template <typename T, DataType dataType>
Element GetCurStartElement(Element start, Element step, int id)
{
    T startValue;
    T stepValue;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        startValue = start.GetSignedData();
        stepValue = step.GetSignedData();
    } else if (dataType == DT_FP32) {
        startValue = (float)start.GetFloatData();
        stepValue = (float)step.GetFloatData();
    }
    T curStartValue = startValue + id * stepValue;
    Element curStart(dataType, curStartValue);
    return curStart;
}

const double EPSILON = (double)1e-12;
template <typename T, DataType dataType>
int64_t GetRangeResSize(Element& start, Element& end, Element& step)
{
    int64_t resultSize;
    if (dataType == DT_INT32 || dataType == DT_INT64) {
        int64_t startValue = start.GetSignedData();
        int64_t endValue = end.GetSignedData();
        int64_t stepValue = step.GetSignedData();
        if (abs(stepValue) <= 0) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "stepValue must not be 0";
        }
        resultSize = (endValue - startValue) % stepValue ? (endValue - startValue) / stepValue + 1 :
                                                           (endValue - startValue) / stepValue;
    } else if (dataType == DT_FP32) {
        double startValue = start.GetFloatData();
        double endValue = end.GetFloatData();
        double stepValue = step.GetFloatData();
        if (abs(stepValue) <= EPSILON) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "stepValue must not be 0";
        }
        resultSize = static_cast<int64_t>(std::ceil((endValue - startValue) / stepValue));
    }
    return resultSize;
}

void TiledRange(Function& function, const TileShape& tileShape, const Element start, const Element step,
                const LogicalTensorPtr& result)
{
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < result->shape[0]; i += vecTile[0]) {
        resultTileInfo.offset[0] = i;
        resultTileInfo.shape[0] = std::min(result->shape[0] - resultTileInfo.offset[0], vecTile[0]);
        int64_t curSizeValue = resultTileInfo.shape[0];
        Element curSize(DT_INT64, curSizeValue);
        Element curStart = start;

        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_RANGE, {}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "START", curStart);
        op.SetAttribute(OP_ATTR_PREFIX + "SIZE", curSize);
        op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
        SymbolicScalar tileIdx(i);
        op.SetAttribute(OpAttributeKey::dynScalar, tileIdx);
    }
    return;
}

LogicalTensorPtr TensorRange(Function& function, LogicalTensorPtr& result, Element& start, Element& step)
{
    auto& op = function.AddOperation(Opcode::OP_RANGE, {}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "START", start);
    op.SetAttribute(OP_ATTR_PREFIX + "STEP", step);
    Element size(DT_INT64, result->shape[0]);
    op.SetAttribute(OP_ATTR_PREFIX + "SIZE", size);
    return result;
}

Tensor RealRange(Element& start, Element& end, Element& step)
{
    DECLARE_TRACER();
    std::vector<int64_t> resTensorShape;
    int64_t resultSize;
    if (start.GetDataType() == DT_INT32) {
        resultSize = GetRangeResSize<int32_t, DT_INT32>(start, end, step);
    } else if (start.GetDataType() == DT_INT64) {
        resultSize = GetRangeResSize<int64_t, DT_INT64>(start, end, step);
    } else if (start.GetDataType() == DT_FP32) {
        resultSize = GetRangeResSize<float, DT_FP32>(start, end, step);
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported DataType " << DataType2String(start.GetDataType());
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, resultSize > 0)
        << "The positivity or negativity of the step should be aligned with the end-start";
    resTensorShape.push_back(resultSize);
    auto resTensor = Tensor(start.GetDataType(), resTensorShape);
    RETURN_CALL(Range, *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(), start, step);
}

bool IsDataTypeUnsupport(DataType dType)
{
    return dType != DT_FP32 && dType != DT_INT64 && dType != DT_INT32 && dType != DT_FP16 && dType != DT_BF16 &&
           dType != DT_INT16;
}

DataType GetComputeDataType(const Element& start, const Element& end, const Element& step)
{
    DataType startType = start.GetDataType();
    DataType endType = end.GetDataType();
    DataType stepType = step.GetDataType();
    if (IsDataTypeUnsupport(startType)) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported Start DataType " << DataType2String(startType);
    }
    if (IsDataTypeUnsupport(endType)) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported End DataType " << DataType2String(endType);
    }
    if (IsDataTypeUnsupport(stepType)) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported Step DataType " << DataType2String(stepType);
    }
    bool startIsFloat = (startType == DT_FP32 || startType == DT_FP16 || startType == DT_BF16);
    bool endIsFloat = (endType == DT_FP32 || endType == DT_FP16 || endType == DT_BF16);
    bool stepIsFloat = (stepType == DT_FP32 || stepType == DT_FP16 || stepType == DT_BF16);
    if (startIsFloat || endIsFloat || stepIsFloat) {
        return DT_FP32;
    }
    if (SupportsInt64Range() && (startType == DT_INT64 || endType == DT_INT64 || stepType == DT_INT64)) {
        return DT_INT64;
    }
    int64_t startValue = start.GetSignedData();
    int64_t endValue = end.GetSignedData();
    int64_t stepValue = step.GetSignedData();
    bool startFlag = startValue <= INT_MAX && startValue >= INT_MIN;
    bool endFlag = endValue <= INT_MAX && endValue >= INT_MIN;
    bool stepFlag = stepValue <= INT_MAX && stepValue >= INT_MIN;
    if (startFlag && endFlag && stepFlag) {
        return DT_INT32;
    }
    return DT_INT64;
}

DataType GetOutputDataType(const Element& start, const Element& end, const Element& step)
{
    DataType startType = start.GetDataType();
    DataType endType = end.GetDataType();
    DataType stepType = step.GetDataType();
    if (startType == DT_INT16 || endType == DT_INT16 || stepType == DT_INT16) {
        return DT_INT16;
    }
    if (startType == DT_FP32 || endType == DT_FP32 || stepType == DT_FP32) {
        return DT_FP32;
    }
    if (startType == DT_FP16 || endType == DT_FP16 || stepType == DT_FP16) {
        return DT_FP16;
    }
    if (startType == DT_BF16 || endType == DT_BF16 || stepType == DT_BF16) {
        return DT_BF16;
    }
    if (SupportsInt64Range() && (startType == DT_INT64 || endType == DT_INT64 || stepType == DT_INT64)) {
        return DT_INT64;
    }
    return DT_INT32;
}

Element GetElementWithDataType(const Element& element, DataType dataType)
{
    DataType elementType = element.GetDataType();
    bool elementIsFloat = (elementType == DT_FP32) || (elementType == DT_FP16) || (elementType == DT_BF16);
    if (elementIsFloat && dataType == DT_FP32) {
        return Element(dataType, element.GetFloatData());
    } else if (elementIsFloat && dataType != DT_FP32) {
        return Element(dataType, (int64_t)element.GetFloatData());
    } else if (!elementIsFloat && dataType == DT_FP32) {
        return Element(dataType, (double)element.GetSignedData());
    }
    return Element(dataType, element.GetSignedData());
}

Tensor Range(const Element& start, const Element& end, const Element& step)
{
    DataType dataType = GetComputeDataType(start, end, step);
    if (dataType != DT_FP32 && dataType != DT_INT32 && dataType != DT_INT64) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
            << "Unsupported Output DataType " << DataType2String(dataType);
    }
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dataType != DT_INT64 || SupportsInt64Range())
        << "RANGE: DT_INT64 is only supported on DAV_3510 and DAV_9201 architectures.";
    DataType outputDataType = DT_INT32;
    outputDataType = GetOutputDataType(start, end, step);

    Element realStart = GetElementWithDataType(start, dataType);
    Element realEnd = GetElementWithDataType(end, dataType);
    Element realStep = GetElementWithDataType(step, dataType);
    auto resTensor = RealRange(realStart, realEnd, realStep);
    if (outputDataType == DT_BF16) {
        return Cast(resTensor, DT_BF16);
    }
    if (outputDataType == DT_FP16) {
        return Cast(resTensor, DT_FP16);
    }
    if (outputDataType == DT_INT16) {
        return Cast(resTensor, DT_INT16);
    }
    return resTensor;
}

void RangeOperationTileFunc(Function& function, const TileShape& tileShape,
                            [[maybe_unused]] const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    Element start = op.GetElementAttribute(OP_ATTR_PREFIX + "START");
    Element step = op.GetElementAttribute(OP_ATTR_PREFIX + "STEP");
    TiledRange(function, tileShape, start, step, oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_RANGE, Opcode::OP_RANGE, RangeOperationTileFunc);

} // namespace npu::tile_fwk
