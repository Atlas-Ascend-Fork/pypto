/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file operation_common.cpp
 * \brief
 */

#include "operation_common.h"
#include "interface/function/function.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "tilefwk/error_code.h"
#include "interface/utils/string_utils.h"
#include "tilefwk/element.h"
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace npu::tile_fwk {

void CheckTensorDynamicShape(const LogicalTensors iOperands, const Opcode opCode)
{
    const std::string opName = OpcodeManager::Inst().GetOpcodeStr(opCode);
    const auto& inputMemType = OpcodeManager::Inst().GetInputsMemType(opCode);
    if (inputMemType.size() != iOperands.size()) {
        return;
    }
    // On Kirin (LiteNPU) the litenpu_dynamic_to_static pass resolves -1 (dynamic) dims to
    // concrete shapes, so -1 is temporarily allowed as an operation operand; other platforms
    // must resolve dynamic shapes via view in pypto.loop before computation.
    const bool allowDynamicShape = IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch());
    for (size_t i = 0; i < iOperands.size(); i++) {
        CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, iOperands[i] != nullptr)
            << opName << ": Input operand[" << i << "] is nullptr.";
        if (inputMemType[i] == MemoryType::MEM_DEVICE_DDR ||
            ((opCode == Opcode::OP_SCATTER_INPLACE || opCode == Opcode::OP_SCATTER_ELEMENT_INPLACE) &&
             inputMemType[i] == MemoryType::MEM_UNKNOWN &&
             iOperands[i]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
            continue;
        }
        for (size_t dimIdx = 0; dimIdx < iOperands[i]->shape.size(); ++dimIdx) {
            CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED,
                  iOperands[i]->shape[dimIdx] > 0 || (allowDynamicShape && iOperands[i]->shape[dimIdx] == -1))
                << (!opName.empty() ? "Operation: " + opName : "")
                << " Input operand (name: " << iOperands[i]->tensor->GetSymbol() << ") "
                << " at dimension[" << dimIdx << "] has invalid shape value: " << iOperands[i]->shape[dimIdx]
                << ". Shape dim must be a positive integer; on non-Kirin platforms, -1 (dynamic) must be resolved "
                << "to a static shape via view in pypto.loop before this operation.";
        }
    }
}

std::vector<int> GetBroadCastShape(LogicalTensorPtr& operand1, LogicalTensorPtr& operand2)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, operand1 != nullptr) << "operand1 is nullptr.";
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, operand2 != nullptr) << "operand2 is nullptr.";
    std::vector<int64_t> opShape1(operand1->shape);
    std::vector<int64_t> opShape2(operand2->shape);
    auto maxShapeSize = std::max(opShape1.size(), opShape2.size());
    if (opShape1.size() != maxShapeSize) {
        opShape1.insert(opShape1.begin(), maxShapeSize - opShape1.size(), 1);
    }
    if (opShape2.size() != maxShapeSize) {
        opShape2.insert(opShape2.begin(), maxShapeSize - opShape2.size(), 1);
    }
    std::vector<int> broadCastShape(maxShapeSize, 0);
    for (size_t i = 0; i < maxShapeSize; i++) {
        broadCastShape[i] = std::max(opShape1[i], opShape2[i]);
    }
    return broadCastShape;
}

std::vector<int> GetBroadcastAxes(const Shape& shape1, const Shape& shape2)
{
    Shape shape1_(shape1), shape2_(shape2);
    std::vector<int> result = {};
    auto maxShapeSize = std::max(shape1_.size(), shape2_.size());
    if (shape1_.size() != maxShapeSize) {
        shape1_.insert(shape1_.begin(), maxShapeSize - shape1_.size(), 1);
    }
    if (shape2_.size() != maxShapeSize) {
        shape2_.insert(shape2_.begin(), maxShapeSize - shape2_.size(), 1);
    }
    for (size_t i = 0; i < shape1_.size(); i++) {
        if (shape1_[i] != shape2_[i] && (shape1_[i] == 1 || shape2_[i] == 1)) {
            result.push_back(i);
        }
    }
    return result;
}

void CheckAxisRange(const Tensor& tensor, int& axis)
{
    int shapeSize = tensor.GetShape().size();
    if (axis < 0) {
        axis += shapeSize;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < shapeSize) << "Axis is not in the reasonable range!";
}

void CheckTensorDimRange(const LogicalTensorPtr& tensor, size_t minDim, size_t maxDim, const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor != nullptr) << opName << ": tensor is nullptr.";
    auto shape = tensor->shape;
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, shape.size() >= minDim && shape.size() <= maxDim)
        << "The dims of tensor is out of range [" << minDim << ", " << maxDim << "]"
        << ", actual dims: " << shape.size() << " for op: " << opName;
}

void CheckDstShapeDimRange(const std::vector<int64_t>& shape, size_t minDim, size_t maxDim, const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, shape.size() >= minDim && shape.size() <= maxDim)
        << "The dims of dst shape is out of range [" << minDim << ", " << maxDim << "]"
        << ", actual dims: " << shape.size() << " for op: " << opName;
}

void CheckTensorsDimConsistency(const std::vector<LogicalTensorPtr>& tensors, const std::string& opName)
{
    if (tensors.empty()) {
        return;
    }
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[0] != nullptr) << opName << ": tensors[0] is nullptr.";
    auto firstDim = tensors[0]->shape.size();
    for (size_t i = 1; i < tensors.size(); ++i) {
        CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[i] != nullptr)
            << opName << ": tensors[" << i << "] is nullptr.";
        CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, tensors[i]->shape.size() == firstDim)
            << "Tensor dim inconsistent, tensor[0] dim: " << firstDim << ", tensor[" << i
            << "] dim: " << tensors[i]->shape.size() << " for op: " << opName;
    }
}

void CheckTensorShapeSize(const LogicalTensorPtr& tensor, const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor != nullptr) << opName << ": tensor is nullptr.";
    auto shape = tensor->shape;
    int64_t shapeSize = 1;
    for (const auto& value : shape) {
        if (value > INT32_MAX) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "The dim value of tensor must be less than or equal to INT32_MAX(2,147,483,647), "
                << "actual dim value: " << value << " for op: " << opName;
        }
        if (value > 0) {
            shapeSize *= value;
        }
        if (shapeSize > INT32_MAX) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "The shape size of tensor must be less than or equal to INT32_MAX(2,147,483,647), "
                << "actual shape size: " << shapeSize << " for op: " << opName;
        }
    }
}

void CheckTensorNonEmpty(const LogicalTensorPtr& tensor, const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor != nullptr) << opName << ": tensor is nullptr.";
    auto shape = tensor->shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape[i] != 0)
            << "Empty tensor is not supported for op: " << opName << ", dimension[" << i << "] has shape value "
            << shape[i] << ".";
    }
}

void CheckVectorOperand(const LogicalTensorPtr& tensor, const std::string& opName,
                        const std::unordered_set<TileOpFormat>& unsupportedFormats)
{
    CheckTensorFormat(tensor, unsupportedFormats, opName);
    CheckTensorNonEmpty(tensor, opName);
    CheckTensorShapeSize(tensor, opName);
}

void CheckDstShapeSize(const std::vector<int64_t>& shape, const std::string& opName)
{
    int64_t shapeSize = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        const auto& value = shape[i];
        if (value > INT32_MAX) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "The dim value of dst shape must be less than or equal to INT32_MAX(2,147,483,647), "
                << "actual dim value: " << value << " for op: " << opName;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, value != 0)
            << "Empty tensor is not supported for op: " << opName << ", dimension[" << i << "] has shape value "
            << value << ".";
        if (value > 0) {
            shapeSize *= value;
        }
        if (shapeSize > INT32_MAX) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "The shape size of dst must be less than or equal to INT32_MAX(2,147,483,647), "
                << "actual shape size: " << shapeSize << " for op: " << opName;
        }
    }
}

bool IsShapeConsistentOrBroadcastCompatible(const Shape& shape1, const Shape& shape2)
{
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
            return false;
        }
    }
    return true;
}

void CheckTensorsShapeConsistencyOrBroadcast(const std::vector<LogicalTensorPtr>& tensors, const std::string& opName)
{
    if (tensors.empty()) {
        return;
    }
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[0] != nullptr) << opName << ": tensors[0] is nullptr.";
    for (size_t i = 1; i < tensors.size(); ++i) {
        CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[i] != nullptr)
            << opName << ": tensors[" << i << "] is nullptr.";
        Shape shape0 = tensors[0]->shape;
        Shape shapeI = tensors[i]->shape;
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              shape0 == shapeI || IsShapeConsistentOrBroadcastCompatible(shape0, shapeI))
            << "Tensor shape must be consistent or broadcast compatible"
            << ", tensor[0] shape: " << StringUtils::ToString(shape0) << ", tensor[" << i
            << "] shape: " << StringUtils::ToString(shapeI) << " for op: " << opName;
    }
}

void CheckTensorDataType(DataType dtype, const std::unordered_set<DataType>& supportedTypes, const std::string& opName)
{
    bool isSupported = supportedTypes.find(dtype) != supportedTypes.end();
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, isSupported)
        << "Data type " << DataType2String(dtype) << " is not in supported types for op: " << opName;
}

void CheckTensorDataType(const LogicalTensorPtr& tensor, const std::unordered_set<DataType>& supportedTypes,
                         const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor != nullptr) << opName << ": tensor is nullptr.";
    auto dtype = tensor->Datatype();
    CheckTensorDataType(dtype, supportedTypes, opName);
}

void CheckTensorDataTypeCombination(const std::vector<DataType>& actualTypes,
                                    const std::vector<std::vector<DataType>>& combinations, const std::string& opName)
{
    for (const auto& combination : combinations) {
        if (combination.size() != actualTypes.size()) {
            continue;
        }
        bool matched = true;
        for (size_t i = 0; i < combination.size(); ++i) {
            if (combination[i] == DataType::DT_BOTTOM) {
                continue;
            }
            if (combination[i] != actualTypes[i]) {
                matched = false;
                break;
            }
        }
        if (matched) {
            return;
        }
    }
    std::ostringstream oss;
    for (size_t i = 0; i < actualTypes.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << DataType2String(actualTypes[i]);
    }
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
        << "Data type combination (" << oss.str() << ") is not supported for op: " << opName;
}

void CheckTensorFormat(const LogicalTensorPtr& tensor, const std::unordered_set<TileOpFormat>& unsupportedFormats,
                       const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor != nullptr) << opName << ": tensor is nullptr.";
    auto format = tensor->Format();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, unsupportedFormats.find(format) == unsupportedFormats.end())
        << "Tensor format " << std::to_string(format) << " is not supported for op: " << opName;
}

void CheckSupportedNPUArch(const std::vector<NPUArch>& supportedArches, const std::string& opName)
{
    if (supportedArches.empty()) {
        return;
    }

    auto arch = Platform::Instance().GetSoc().GetNPUArch();
    bool isSupported = std::find(supportedArches.begin(), supportedArches.end(), arch) != supportedArches.end();
    std::ostringstream oss;
    for (size_t i = 0; i < supportedArches.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << NPUArchToString(supportedArches[i]);
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, isSupported)
        << opName << ": this interface only supports architecture: " << oss.str();
}

bool IsVectorOpSimtEnabled(Opcode opcode)
{
    static const std::unordered_map<NPUArch, std::unordered_set<Opcode>> simtOpsByArch = {
        {NPUArch::DAV_1001, {}},
        {NPUArch::DAV_2201, {}},
        {NPUArch::DAV_3510,
         {Opcode::OP_GCD, Opcode::OP_GCDS, Opcode::OP_INDEX_ADD, Opcode::OP_INDEX_PUT, Opcode::OP_SCATTER_INPLACE,
          Opcode::OP_SCATTER_ELEMENT_INPLACE}},
        {NPUArch::DAV_9201,
         {Opcode::OP_GCD, Opcode::OP_GCDS, Opcode::OP_INDEX_ADD, Opcode::OP_INDEX_PUT, Opcode::OP_SCATTER_INPLACE,
          Opcode::OP_SCATTER_ELEMENT_INPLACE}},
        {NPUArch::DAV_3003, {}},
        {NPUArch::DAV_3113, {}},
        {NPUArch::DAV_5101, {}},
    };
    auto arch = Platform::Instance().GetSoc().GetNPUArch();
    auto it = simtOpsByArch.find(arch);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, it != simtOpsByArch.end())
        << "Vector SIMT selection is not configured for architecture: " << NPUArchToString(arch);
    return it->second.count(opcode) != 0;
}

void CheckTensorsDataTypeConsistency(const LogicalTensorPtr& tensor1, const LogicalTensorPtr& tensor2,
                                     const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor1 != nullptr) << opName << ": tensor1 is nullptr.";
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor2 != nullptr) << opName << ": tensor2 is nullptr.";
    auto dtype1 = tensor1->Datatype();
    auto dtype2 = tensor2->Datatype();
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype1 == dtype2)
        << "Tensor data type inconsistent, tensor1 dtype: " << DataType2String(dtype1)
        << ", tensor2 dtype: " << DataType2String(dtype2) << " for op: " << opName;
}

void CheckTensorsDataTypeConsistency(const LogicalTensorPtr& tensor, const Element& element, const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor != nullptr) << opName << ": tensor is nullptr.";
    auto dtype1 = tensor->Datatype();
    auto dtype2 = element.GetDataType();
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype1 == dtype2)
        << "Tensor and Element data type inconsistent, tensor dtype: " << DataType2String(dtype1)
        << ", element dtype: " << DataType2String(dtype2) << " for op: " << opName;
}

void CheckTensorsDataTypeConsistency(const std::vector<LogicalTensorPtr>& tensors, const std::string& opName)
{
    if (tensors.empty()) {
        return;
    }
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[0] != nullptr) << opName << ": tensors[0] is nullptr.";
    auto firstDtype = tensors[0]->Datatype();
    for (size_t i = 1; i < tensors.size(); ++i) {
        CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[i] != nullptr)
            << opName << ": tensors[" << i << "] is nullptr.";
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, tensors[i]->Datatype() == firstDtype)
            << "Tensor data type inconsistent, tensor0 dtype: " << DataType2String(firstDtype) << ", tensor" << i
            << " dtype: " << DataType2String(tensors[i]->Datatype()) << " for op: " << opName;
    }
}

void CheckTensorsFormatConsistency(const LogicalTensorPtr& tensor1, const LogicalTensorPtr& tensor2,
                                   const std::string& opName)
{
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor1 != nullptr) << opName << ": tensor1 is nullptr.";
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensor2 != nullptr) << opName << ": tensor2 is nullptr.";
    auto format1 = tensor1->Format();
    auto format2 = tensor2->Format();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, format1 == format2)
        << "Tensor format inconsistent, tensor1 format: " << std::to_string(format1)
        << ", tensor2 format: " << std::to_string(format2) << " for op: " << opName;
}

void CheckTensorsFormatConsistency(const std::vector<LogicalTensorPtr>& tensors, const std::string& opName)
{
    if (tensors.empty()) {
        return;
    }
    CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[0] != nullptr) << opName << ": tensors[0] is nullptr.";
    auto firstFormat = tensors[0]->Format();
    for (size_t i = 1; i < tensors.size(); ++i) {
        CHECK(VectorErrorCode::ERR_RUNTIME_NULLPTR, tensors[i] != nullptr)
            << opName << ": tensors[" << i << "] is nullptr.";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, tensors[i]->Format() == firstFormat)
            << "Tensor format inconsistent, tensor0 format: " << std::to_string(firstFormat) << ", tensor" << i
            << " format: " << std::to_string(tensors[i]->Format()) << " for op: " << opName;
    }
}

void CheckBinaryInputTensors(const LogicalTensorPtr& tensor1, const LogicalTensorPtr& tensor2,
                             const std::string& opName)
{
    // Check tensor1's properties
    CheckTensorDimRange(tensor1, MIN_TENSOR_DIM, MAX_TENSOR_DIM, opName);
    CheckTensorShapeSize(tensor1, opName);

    // Check tensor2's shape size only (other properties will be checked via consistency)
    CheckTensorShapeSize(tensor2, opName);

    // Check tensors consistency
    CheckTensorsDimConsistency({tensor1, tensor2}, opName);
    CheckTensorsShapeConsistencyOrBroadcast({tensor1, tensor2}, opName);
    CheckTensorsFormatConsistency(tensor1, tensor2, opName);
}

static LogicalTensorPtr MakeNanScratch(Function& function, size_t rank, size_t bytes)
{
    Shape shape(rank, 1);
    shape.back() = bytes;
    return std::make_shared<LogicalTensor>(function, DT_UINT8, shape, SymbolicScalar::FromConcrete(shape));
}

static size_t NanReduceScratchBytes(const Shape& tile, DataType dtype, size_t axis, bool arg)
{
    const bool column = axis + 1 != tile.size();
    const size_t e = BytesOf(dtype);
    const size_t h = column ? tile[axis] : (tile.size() > 1 ? tile[tile.size() - 2] : 1);
    size_t w = tile.back();
    if (column) {
        for (size_t i = axis + 1; i + 1 < tile.size(); ++i) {
            w *= tile[i];
        }
    }
    const size_t rowBytes = AlignUp(w * e, size_t{BLOCK_SIZE});
    const size_t dataBytes = h * rowBytes;
    // The final vector compare can read beyond the last physical row.
    const size_t compareGuard = AlignUp(rowBytes, REPEAT_BYTE) - rowBytes;
    const size_t maskBytes = h * AlignUp((rowBytes / e + 7) / 8, size_t{BLOCK_SIZE});
    const size_t flagRowBytes = column ? rowBytes : BLOCK_SIZE;
    const size_t resultBytes = column ? rowBytes : h * BLOCK_SIZE;
    // Arg candidates retain a compact input copy, including FP32 inputs.
    const size_t castBytes = arg ? dataBytes : 0;
    const size_t rowReduceBytes = rowBytes <= REPEAT_BYTE ?
                                      (arg ? h * BLOCK_SIZE : BLOCK_SIZE) :
                                      h * (rowBytes <= 2 * REPEAT_BYTE ? REPEAT_BYTE :
                                                                         rowBytes / (2 * REPEAT_BYTE) * REPEAT_BYTE);
    const size_t reduceBytes = column ? (arg ? 3 * REPEAT_BYTE : 0) : rowReduceBytes;
    const size_t flagEnd = castBytes + dataBytes + compareGuard + maskBytes + (column ? resultBytes : 2 * resultBytes);
    const size_t flagReadEnd = flagEnd + AlignUp(flagRowBytes, size_t{REPEAT_BYTE}) - flagRowBytes;
    const size_t selectorEnd = flagEnd + (arg && !column ? resultBytes : 0) + reduceBytes + BLOCK_SIZE;
    return std::max(flagReadEnd, selectorEnd);
}

static size_t NanSmallScratchBytes(const LogicalTensorPtr& tile, bool pairArg)
{
    const auto& shape = tile->shape;
    const size_t w = AlignUp(shape.back(), static_cast<int64_t>(BLOCK_SIZE / BytesOf(tile->Datatype())));
    const size_t h = shape.size() > 1 ? shape[shape.size() - 2] : 1;
    const size_t maskBytes = h * AlignUp((w + 7) / 8, size_t{BLOCK_SIZE});
    if (!pairArg) {
        // A private, initialized full-repeat copy also protects runtime partial tails.
        const size_t compareBytes = h * AlignUp(w * BytesOf(tile->Datatype()), size_t{REPEAT_BYTE});
        return maskBytes + BLOCK_SIZE + compareBytes;
    }
    const size_t indexBytes = h * AlignUp(w * sizeof(int32_t), size_t{BLOCK_SIZE});
    const size_t rowBytes = w * sizeof(int32_t);
    const size_t guard = AlignUp(rowBytes, size_t{REPEAT_BYTE}) - rowBytes;
    const size_t compareBytes = rowBytes > REPEAT_BYTE && guard != 0 ? 2 * (h * rowBytes + guard) : 0;
    return maskBytes + indexBytes + BLOCK_SIZE + compareBytes;
}

bool IgnoreNanForCurrentTileOp()
{
    auto* root = ExpandFunction::GetCurrentTileOp();
    return root != nullptr && root->HasAttr(OP_ATTR_PREFIX + "propagate_nan") &&
           !root->GetBoolAttribute(OP_ATTR_PREFIX + "propagate_nan");
}

Operation& AddNanReduceOperation(Function& function, const std::string& name, LogicalTensors inputs,
                                 LogicalTensors outputs, size_t axis)
{
    const auto code = FindOpcode(name);
    const bool arg = code == Opcode::OP_ROWARGMAXWITHVALUE_SINGLE || code == Opcode::OP_ROWARGMINWITHVALUE_SINGLE ||
                     code == Opcode::OP_ROWARGMAXWITHVALUE_LINE || code == Opcode::OP_ROWARGMINWITHVALUE_LINE;
    if (!IgnoreNanForCurrentTileOp() || !IsFloat(inputs[0]->Datatype())) {
        return function.AddOperation(code, inputs, outputs);
    }
    const auto dtype = inputs[0]->Datatype();
    auto shape = inputs[0]->shape;
    const auto align = static_cast<int64_t>(BLOCK_SIZE / BytesOf(dtype));
    shape.back() = AlignUp(shape.back(), align);
    // Standard tail padding also covers the workspace of a combined singleton tail.
    const auto bytes = NanReduceScratchBytes(shape, dtype, axis, arg);
    auto scratch = MakeNanScratch(function, shape.size(), bytes);
    // Replace the original native scratch, rather than allocating both.
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, outputs.size() == (arg ? 2U : 1U))
        << "NaN reduction expects " << (arg ? 2 : 1) << " outputs, but got " << outputs.size();
    outputs.push_back(scratch);
    auto& result = function.AddOperation(code, inputs, outputs);
    result.SetAttribute(OP_ATTR_PREFIX + "nan_tile", true);
    result.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
    return result;
}

Operation& AddNanSmallOperation(Function& function, Opcode code, LogicalTensors inputs, LogicalTensors outputs,
                                bool finalize)
{
    const bool pairArg = code == Opcode::OP_PAIRARGMAX || code == Opcode::OP_PAIRARGMIN;
    if (!IgnoreNanForCurrentTileOp() || (!pairArg && !IsFloat(inputs[0]->Datatype())))
        return function.AddOperation(code, inputs, outputs);
    auto tmp = MakeNanScratch(function, outputs[0]->shape.size(), NanSmallScratchBytes(outputs[0], pairArg));
    outputs.push_back(tmp);
    auto& result = function.AddOperation(code, inputs, outputs);
    result.SetAttribute(OP_ATTR_PREFIX + "nan_tile", true);
    if (pairArg) {
        // One flag per input operand: the PAIR ops keep their non-overlapping tail.
        // SetAttr (not SetAttribute) because std::vector<bool> cannot be iterated by reference.
        result.SetAttr(OpAttributeKey::rowPad, std::vector<bool>(inputs.size(), true));
        if (finalize) {
            result.SetAttribute(OP_ATTR_PREFIX + "nan_finalize", true);
        }
    }
    result.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
    return result;
}
} // namespace npu::tile_fwk
