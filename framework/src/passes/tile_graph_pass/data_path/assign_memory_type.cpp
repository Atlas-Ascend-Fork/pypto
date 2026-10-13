/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file assign_memory_type.cpp
 * \brief
 */

#include "assign_memory_type.h"

#include <algorithm>
#include <set>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/simt_utils.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/checker_utils.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/tile_graph_pass/data_path/memory_path_utils.h"
#include "tilefwk/tilefwk.h"

#define MODULE_NAME "AssignMemoryType"

#define RETURN_IF_NOT_SUCCESS(expr)                \
    do {                                           \
        Status assignMemoryReturnStatus = (expr);  \
        if (assignMemoryReturnStatus != SUCCESS) { \
            return assignMemoryReturnStatus;       \
        }                                          \
    } while (0)

namespace npu::tile_fwk {
Status AssignMemoryType::RunOnFunction(Function& function)
{
    mode_ = config::EnableSlice() ? AssignMemoryTypeMode::SLICE_CONTRACT : AssignMemoryTypeMode::VIEW_ASSEMBLE;
    inserter = ConvertInserter(mode_);
    APASS_LOG_INFO_F(Elements::Function, "===> Start AssignMemoryType.");
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    if (IsSliceMode()) {
        RETURN_IF_NOT_SUCCESS(AssignConfirmedMemoryTypes(function));
        RETURN_IF_NOT_SUCCESS(InferUncertainMemoryTypes(function));
    } else {
        RETURN_IF_NOT_SUCCESS(AssignConfirmedMemoryTypesLegacy(function));
        RETURN_IF_NOT_SUCCESS(InferUncertainMemoryTypesLegacy(function));
    }
    RETURN_IF_NOT_SUCCESS(ResolveMemoryUnknowns(function));
    RETURN_IF_NOT_SUCCESS(ResolveInconsistentRawTensorMemoryTypes(function));
    RETURN_IF_NOT_SUCCESS(SyncViewAssembleMemoryAttrs(function));
    RETURN_IF_NOT_SUCCESS(FixViewAssembleSemanticMismatch(function));
    RETURN_IF_NOT_SUCCESS(ResolveUnalignedUbSlices(function));
    RETURN_IF_NOT_SUCCESS(InsertConvertOpsAndInferShape(function));
    RETURN_IF_NOT_SUCCESS(NormalizeViewAssembleOpcodes(function));
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SyncTensorToBe(function));
    RETURN_IF_NOT_SUCCESS(MarkA5SimtGatherElement(function));
    APASS_LOG_INFO_F(Elements::Function, "===> End AssignMemoryType.");
    return SUCCESS;
}

Status AssignMemoryType::AssignConfirmedMemoryTypes(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW || op.GetOpcode() == Opcode::OP_SLICE) {
            RETURN_IF_NOT_SUCCESS(AssignViewAttrMemoryType(op));
            RETURN_IF_NOT_SUCCESS(AssignSliceInputRequirement(op));
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_CONTRACT) {
            RETURN_IF_NOT_SUCCESS(AssignAssembleAttrMemoryType(op));
        }
        if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignReduceAccInputRequirements(inserter, op));
        }
        if (OpChecker::check(op, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
            RETURN_IF_NOT_SUCCESS(AssignMatmulInputRequirements(op));
        }
        RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignOpcodeDefinedMemoryTypes(inserter, op));
        // shape 存在 -1（动态维度）的输出 tensor 无法作为本地 buffer 静态分配, 统一绑定 DDR
        RETURN_IF_NOT_SUCCESS(AssignDynamicShapeOutputMemoryType(op));
    }
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::AssignInOutCastMemoryTypes(function));
    return MemoryPathUtils::EnsureAllConsumerRequirementsExist(inserter, function);
}

Status AssignMemoryType::AssignDynamicShapeOutputMemoryType(Operation& operation)
{
    for (auto& output : operation.oOperand) {
        const auto& outShape = output->GetShape();
        if (std::any_of(outShape.begin(), outShape.end(), [](int64_t dim) { return dim < 0; })) {
            MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "DynamicShapeTensorAsDdr");
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignMatmulInputRequirements(Operation& operation)
{
    for (auto& tensor : operation.iOperand) {
        for (const auto& producerOp : tensor->GetProducers()) {
            auto producerOpcode = producerOp->GetOpcode();
            MemoryType requirement = MemoryType::MEM_DEVICE_DDR;
            if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MATMUL))) {
                requirement = MemoryType::MEM_L0C;
            } else if (IsModeMoveIn(mode_, producerOpcode)) {
                auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
                if (viewOpAttribute == nullptr) {
                    APASS_LOG_ERROR_F(Elements::Operation,
                                      "View attribute is null for %s[%d] while assigning matmul input.",
                                      producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic());
                    return FAILED;
                }
                requirement = viewOpAttribute->GetTo();
                if (requirement == MemoryType::MEM_UNKNOWN) {
                    requirement = MemoryType::MEM_DEVICE_DDR;
                }
            } else if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                        OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                        OpChecker::OutputMemTypeChecker(MemoryType::MEM_L0A))) {
                requirement = MemoryType::MEM_L0A;
            } else if (OpChecker::check(producerOp, OpChecker::CalcTypeChecker(OpCalcType::MOVE_LOCAL),
                                        OpChecker::InputMemTypeChecker(MemoryType::MEM_L1),
                                        OpChecker::OutputMemTypeChecker(MemoryType::MEM_L0B))) {
                requirement = MemoryType::MEM_L0B;
            }
            RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetRequirementChecked(inserter, tensor, operation, requirement,
                                                                         "AssignMatmulInputRequirements"));
            if (requirement != MemoryType::MEM_DEVICE_DDR && requirement != MemoryType::MEM_UNKNOWN) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Infer %s[%d] input tensor[%d] as %s.",
                                  operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), tensor->GetMagic(),
                                  BriefMemoryTypeToString(requirement).c_str());
            }
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignViewAttrMemoryType(Operation& operation)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "View attribute is null for %s[%d] while assigning view attr memory type.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType attrToType = viewOpAttribute->GetTo();
    if (attrToType == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(
        MemoryPathUtils::SetOriginalChecked(operation.oOperand.front(), attrToType, "AssignViewAttrMemoryType"));
    if (operation.GetOpcode() == Opcode::OP_VIEW) {
        RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetRequirementChecked(inserter, operation.iOperand.front(), operation,
                                                                     attrToType, "AssignViewAttrMemoryType"));
    }
    return SUCCESS;
}

Status AssignMemoryType::AssignSliceInputRequirement(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_SLICE) {
        return SUCCESS;
    }
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assign OP_SLICE[%d] input requirement failed because input operand is empty or null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.empty() ? nullptr : operation.oOperand.front();
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assign OP_SLICE[%d] input requirement failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType targetType = viewOpAttribute->GetTo();
    if (output != nullptr && output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        targetType = output->GetMemoryTypeOriginal();
    }
    if (input->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
        (targetType == MemoryType::MEM_L0A || targetType == MemoryType::MEM_L0B)) {
        return MemoryPathUtils::SetRequirementChecked(inserter, input, operation, MemoryType::MEM_L1,
                                                      "AssignSliceInputRequirementLocalCopyIn");
    }
    MemoryType requirement = MemoryType::MEM_DEVICE_DDR;
    for (const auto& producerOp : input->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto producerViewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
        if (producerViewOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(
                Elements::Operation,
                "Assign OP_SLICE[%d] input requirement failed because producer OP_SLICE[%d] view attr is null.",
                operation.GetOpMagic(), producerOp->GetOpMagic());
            return FAILED;
        }
        if (producerViewOpAttribute->GetTo() == MemoryType::MEM_L1) {
            requirement = MemoryType::MEM_L1;
            break;
        }
    }
    return MemoryPathUtils::SetRequirementChecked(inserter, input, operation, requirement,
                                                  "AssignSliceInputRequirement");
}

Status AssignMemoryType::AssignAssembleAttrMemoryType(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_ASSEMBLE && opcode != Opcode::OP_CONTRACT)
        return SUCCESS;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Assemble attribute is null for %s[%d] while assigning assemble attr memory type.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType attrFromType = assembleOpAttribute->GetFrom();
    if (attrFromType == MemoryType::MEM_UNKNOWN)
        return SUCCESS;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetRequirementChecked(inserter, operation.iOperand.front(), operation,
                                                                 attrFromType, "AssignAssembleAttrMemoryType"));
    if (opcode != Opcode::OP_ASSEMBLE) {
        return SUCCESS;
    }
    auto output = operation.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal == MemoryType::MEM_UNKNOWN) {
        return MemoryPathUtils::SetOriginalChecked(output, attrFromType, "AssignAssembleAttrMemoryType");
    }
    if (outputOriginal == attrFromType) {
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation,
                      "Assign assemble attr memory type failed because view operation memory type conflict exists. "
                      "%s[%d] from type is %s, output tensor[%d] original type is %s.",
                      operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                      BriefMemoryTypeToString(attrFromType).c_str(), output->GetMagic(),
                      BriefMemoryTypeToString(outputOriginal).c_str());
    return FAILED;
}

// l0c2ub pattern: batchmatmul case: cube op -> move-out(s) -> reshape op -> move op(s) -> vector
bool AssignMemoryType::IsReshapeCubeToVecL0C2UBPattern(Operation& op)
{
    if (op.GetOpcode() != npu::tile_fwk::Opcode::OP_RESHAPE) {
        return false;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();

    bool isL0C2UBPattern = true;
    if (producers.empty() || consumers.empty()) {
        return false;
    }

    for (auto& producer : producers) {
        bool isProducerMoveOut = IsModeMoveOut(mode_, producer->GetOpcode());
        bool isProducerProducerAllCube = false;

        std::vector<bool> isProducerProducerCube;
        for (auto& producerIOperand : producer->iOperand) {
            for (auto& producerProducer : producerIOperand->GetProducers()) {
                isProducerProducerCube.push_back(producerProducer->GetCoreType() == CoreType::AIC);
            }
        }

        isProducerProducerAllCube = !isProducerProducerCube.empty() &&
                                    std::all_of(isProducerProducerCube.begin(), isProducerProducerCube.end(),
                                                [](bool val) { return val == true; });
        if (!isProducerMoveOut || !isProducerProducerAllCube) {
            isL0C2UBPattern = false;
        }
    }
    for (auto& consumer : consumers) {
        bool isConsumerMoveOp = IsModeMoveOp(mode_, consumer->GetOpcode());
        bool isConsumerConsumerAllVector = false;

        std::vector<bool> isConsumerConsumerVector;
        for (auto& consumerOOperand : consumer->oOperand) {
            for (auto& consumerConsumer : consumerOOperand->GetConsumers()) {
                isConsumerConsumerVector.push_back(consumerConsumer->GetCoreType() == CoreType::AIV);
            }
        }
        isConsumerConsumerAllVector = !isConsumerConsumerVector.empty() &&
                                      std::all_of(isConsumerConsumerVector.begin(), isConsumerConsumerVector.end(),
                                                  [](bool val) { return val == true; });
        if (!isConsumerMoveOp || !isConsumerConsumerAllVector) {
            isL0C2UBPattern = false;
        }
    }
    return isL0C2UBPattern;
}

// ub2l1 pattern: 2 patterns
// 1. vector op -> slice/contract -> reshape op -> slice from l1 -> slice from l0a -> cube
// 2. vector op -> slice/contract -> slice/contract -> reshape op -> slice from l1 -> slice from l0a -> cube
bool AssignMemoryType::IsReshapeVecToCubeUB2L1Pattern(Operation& op)
{
    if (op.GetOpcode() != npu::tile_fwk::Opcode::OP_RESHAPE) {
        return false;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    if ((input == nullptr) || (output == nullptr)) {
        return false;
    }

    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    if (producers.empty() || consumers.empty()) {
        return false;
    }

    if (!IsReshapeVecToCubeUB2L1ProducerPattern(producers)) {
        return false;
    }

    if (!IsReshapeVecToCubeUB2L1ConsumerPattern(consumers)) {
        return false;
    }

    return true;
}

bool AssignMemoryType::IsReshapeVecToCubeUB2L1ProducerPattern(
    const std::set<Operation*, LogicalTensor::CompareOp>& producers)
{
    for (auto& producer : producers) {
        bool isProducerDepth1MoveOp = IsModeMoveOp(mode_, producer->GetOpcode());
        bool isProducerDepth2AllVector = false;
        bool isProducerDepth2AllMoveOp = false;
        bool isProducerDepth3AllVector = false;

        std::vector<bool> isProducerDepth2Vector;
        std::vector<bool> isProducerDepth2MoveOp;
        std::vector<bool> isProducerDepth3Vector;
        for (auto& producerIOperand : producer->iOperand) {
            for (auto& producerProducer : producerIOperand->GetProducers()) {
                isProducerDepth2Vector.push_back(producerProducer->GetCoreType() == CoreType::AIV);
                isProducerDepth2MoveOp.push_back(IsModeMoveOp(mode_, producerProducer->GetOpcode()));
                MemoryPathUtils::CollectProducerAIVFlags(producerProducer, isProducerDepth3Vector);
            }
        }
        isProducerDepth2AllVector = !isProducerDepth2Vector.empty() &&
                                    std::all_of(isProducerDepth2Vector.begin(), isProducerDepth2Vector.end(),
                                                [](bool val) { return val; });
        isProducerDepth2AllMoveOp = !isProducerDepth2MoveOp.empty() &&
                                    std::all_of(isProducerDepth2MoveOp.begin(), isProducerDepth2MoveOp.end(),
                                                [](bool val) { return val; });
        isProducerDepth3AllVector = !isProducerDepth3Vector.empty() &&
                                    std::all_of(isProducerDepth3Vector.begin(), isProducerDepth3Vector.end(),
                                                [](bool val) { return val; });
        // currently only support the following patterns:
        // 1. vector op -> slice(s)/contract(s) -> reshape op
        // 2. vector op -> slice(s)/contract(s) -> slice(s)/contract(s) -> reshape op
        if (!((isProducerDepth1MoveOp && isProducerDepth2AllMoveOp && isProducerDepth3AllVector) ||
              (isProducerDepth1MoveOp && isProducerDepth2AllVector))) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::IsReshapeVecToCubeUB2L1ConsumerPattern(
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    for (auto& consumer : consumers) {
        bool isConsumerDepth1MoveIn = IsModeMoveIn(mode_, consumer->GetOpcode());
        bool isConsumerDepth2AllMoveIn = false;
        bool isConsumerDepth3AllCube = false;

        std::vector<bool> isConsumerDepth2MoveIn;
        std::vector<bool> isConsumerDepth3Cube;
        for (auto& consumerOOperand : consumer->oOperand) {
            for (auto& consumerConsumer : consumerOOperand->GetConsumers()) {
                isConsumerDepth2MoveIn.push_back(IsModeMoveIn(mode_, consumerConsumer->GetOpcode()));
                MemoryPathUtils::CollectConsumerAICFlags(consumerConsumer, isConsumerDepth3Cube);
            }
        }
        isConsumerDepth2AllMoveIn = !isConsumerDepth2MoveIn.empty() &&
                                    std::all_of(isConsumerDepth2MoveIn.begin(), isConsumerDepth2MoveIn.end(),
                                                [](bool val) { return val; });
        isConsumerDepth3AllCube = !isConsumerDepth3Cube.empty() &&
                                  std::all_of(isConsumerDepth3Cube.begin(), isConsumerDepth3Cube.end(),
                                              [](bool val) { return val; });
        if (!isConsumerDepth1MoveIn || !isConsumerDepth2AllMoveIn || !isConsumerDepth3AllCube) {
            return false;
        }
    }
    return true;
}

Status AssignMemoryType::InferReshapeL0C2UBAndUB2L1PatternLiteNPU(Operation& op)
{
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        return SUCCESS;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();

    // l0c2ub pattern: batchmatmul case: cube op -> contract(s) -> reshape op -> slice(s)/contract(s) -> vector
    if (IsReshapeCubeToVecL0C2UBPattern(op) && MemoryPathUtils::FitsTensorInUb(input) &&
        inserter.IsL0C2UbSupportedDtype(input)) {
        for (auto& producer : producers) {
            auto& producerInput = producer->iOperand.front();
            auto& producerOutput = producer->oOperand.front();

            // set producer contract input to L0C
            producerInput->SetMemoryTypeOriginal(MemoryType::MEM_L0C, true);
            inserter.UpdateTensorTobeMap(producerInput, *producer, MemoryType::MEM_L0C);

            // set producer output to be UB
            producerOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(producerOutput, op, MemoryType::MEM_UB);
        }
        // set reshape output to UB
        output->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

        // set all consumer slice/contracts input to UB
        for (auto& consumer : consumers) {
            inserter.UpdateTensorTobeMap(output, *consumer, MemoryType::MEM_UB);
        }
        return SUCCESS;
    }

    // ub2l1 pattern:
    // 1. vector op -> slice(s)/contract(s) -> reshape op -> slice(s) from l1 -> slice(s) from l0a -> cube
    // 2. vector op -> contract(s) -> slice(s) -> reshape op -> slice(s) from l1 -> slice(s) from l0a -> cube
    // MXMatmul场景K轴非64对齐不支持UB2L1直连（MX补齐仅由DDR路径支持），回退DDR
    if (IsReshapeVecToCubeUB2L1Pattern(op) && MemoryPathUtils::FitsTensorInUb(output) &&
        inserter.IsUb2L1SupportedDtype(output) && !MemoryPathUtils::HasMxPaddingModeConsumer(output)) {
        for (auto& producer : producers) {
            auto& producerInput = producer->iOperand.front();
            auto& producerOutput = producer->oOperand.front();

            // set producer slice/contract input to UB
            producerInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(producerInput, *producer, MemoryType::MEM_UB);

            // set reshape input to UB
            producerOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(producerOutput, op, MemoryType::MEM_UB);
        }

        for (auto& consumer : consumers) {
            auto& consumerInput = consumer->iOperand.front();
            auto& consumerOutput = consumer->oOperand.front();

            // set reshape output to UB, set its tobe mem type to L1
            consumerInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            inserter.UpdateTensorTobeMap(consumerInput, *consumer, MemoryType::MEM_L1);

            // set consumer slice output to be L1
            consumerOutput->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
            for (auto& consumerConsumer : consumerOutput->GetConsumers()) {
                inserter.UpdateTensorTobeMap(consumerOutput, *consumerConsumer, MemoryType::MEM_L1);
            }
        }
        return SUCCESS;
    }

    return SUCCESS;
}

Status AssignMemoryType::InferUncertainMemoryTypes(Function& function)
{
    std::unordered_set<LogicalTensorPtr> inferredAssembleOutputs;
    for (auto& op : function.Operations()) {
        switch (op.GetOpcode()) {
            case Opcode::OP_VIEW:
                RETURN_IF_NOT_SUCCESS(InferViewMemoryType(op));
                break;
            case Opcode::OP_SLICE:
                RETURN_IF_NOT_SUCCESS(InferSliceMemoryType(op));
                break;
            case Opcode::OP_VIEW_TYPE:
                RETURN_IF_NOT_SUCCESS(InferViewTypeMemoryType(op));
                break;
            case Opcode::OP_ASSEMBLE:
                RETURN_IF_NOT_SUCCESS(InferAssembleMemoryType(function, op, inferredAssembleOutputs));
                break;
            case Opcode::OP_CONTRACT:
                RETURN_IF_NOT_SUCCESS(InferContractMemoryType(op));
                break;
            case Opcode::OP_RESHAPE:
                RETURN_IF_NOT_SUCCESS(InferReshapeMemoryType(op));
                RETURN_IF_NOT_SUCCESS(InferReshapeL0C2UBAndUB2L1PatternLiteNPU(op));
                break;
            default:
                break;
        }
    }

    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::ApplyOtherSpecialOpcodeRules(inserter, function));
    RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(function));
    return ApplyPlatformPathUpgradeRules(function);
}

static bool IsViewSemanticOpcode(Opcode opcode)
{
    return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_RESHAPE;
}

Status AssignMemoryType::InferViewMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_VIEW, "Infer OP_VIEW memory type", input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryType inferredType = InferRequirementFromInputOriginals(input);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        inferredType = InferOriginalFromOutputRequirements(output);
    }
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, inferredType, "InferViewSemanticType");
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, inferredType, "InferViewSemanticType"));
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute != nullptr) {
        viewOpAttribute->SetToType(inferredType);
    }
    return SUCCESS;
}

Status AssignMemoryType::InferSliceMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_SLICE, "Infer OP_SLICE memory type", input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_SLICE[%d] memory type failed because view attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
        viewOpAttribute->SetToType(output->GetMemoryTypeOriginal());
        return SUCCESS;
    }
    MemoryType inferredType = InferOriginalFromOutputRequirements(output);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        inferredType = MemoryType::MEM_UB;
    }
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, inferredType, "InferSliceOutputOriginal"));
    viewOpAttribute->SetToType(inferredType);
    return SUCCESS;
}

Status AssignMemoryType::InferContractMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_CONTRACT, "Infer OP_CONTRACT memory type", input, output, shouldHandle));
    if (!shouldHandle)
        return SUCCESS;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_CONTRACT[%d] memory type failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
    if (inputRequirement != MemoryType::MEM_UNKNOWN) {
        assembleOpAttribute->SetFromType(inputRequirement);
        return SUCCESS;
    }
    MemoryType inferredType = InferRequirementFromInputOriginals(input);
    if (inferredType == MemoryType::MEM_UNKNOWN) {
        inferredType = MemoryType::MEM_UB;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, inferredType, "InferContractInputRequirement");
    assembleOpAttribute->SetFromType(inferredType);
    return SUCCESS;
}

MemoryType AssignMemoryType::InferOriginalFromOutputRequirements(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    return InferOriginalFromOutputRequirements(tensor, visitedTensors);
}

MemoryType AssignMemoryType::InferOriginalFromOutputRequirements(
    const LogicalTensorPtr& tensor, std::unordered_set<const LogicalTensor*>& visitedTensors) const
{
    if (tensor == nullptr || !visitedTensors.insert(tensor.get()).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    auto addCandidate = [&candidates](MemoryType candidate) {
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    };
    auto consumerRequirements = inserter.GetConsumerRequirements(tensor);
    for (const auto& item : consumerRequirements) {
        Operation* consumerOp = item.first;
        MemoryType requirement = item.second;
        if (requirement != MemoryType::MEM_UNKNOWN) {
            addCandidate(requirement);
            continue;
        }
        if (consumerOp == nullptr || !IsViewSemanticOpcode(consumerOp->GetOpcode())) {
            continue;
        }
        for (const auto& output : consumerOp->oOperand) {
            addCandidate(InferOriginalFromOutputRequirements(output, visitedTensors));
        }
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
}

MemoryType AssignMemoryType::InferRequirementFromInputOriginals(const LogicalTensorPtr& tensor) const
{
    std::unordered_set<const LogicalTensor*> visitedTensors;
    return InferRequirementFromInputOriginals(tensor, visitedTensors);
}

MemoryType AssignMemoryType::InferRequirementFromInputOriginals(
    const LogicalTensorPtr& tensor, std::unordered_set<const LogicalTensor*>& visitedTensors) const
{
    if (tensor == nullptr || !visitedTensors.insert(tensor.get()).second) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    auto addCandidate = [&candidates](MemoryType candidate) {
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    };
    addCandidate(tensor->GetMemoryTypeOriginal());
    for (const auto& producerOp : tensor->GetProducers()) {
        if (producerOp == nullptr || !IsViewSemanticOpcode(producerOp->GetOpcode())) {
            continue;
        }
        for (const auto& input : producerOp->iOperand) {
            addCandidate(InferRequirementFromInputOriginals(input, visitedTensors));
        }
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_UNKNOWN;
}

bool AssignMemoryType::TryHandleSpecialDirectMemoryPath(Operation& operation, MemoryType from, MemoryType to,
                                                        bool& directPath)
{
    LogicalTensorPtr input = operation.iOperand.empty() ? nullptr : operation.iOperand.front();
    if (MemoryPathUtils::IsSpecialDirectMemoryPath(from, to) && HasParallelDifferentConsumerRequirement(input, to)) {
        directPath = false;
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Disable direct %s -> %s path for %s[%d] because source tensor has parallel different requirements.",
            BriefMemoryTypeToString(from).c_str(), BriefMemoryTypeToString(to).c_str(),
            operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return true;
    }
    if (from == MemoryType::MEM_L0C && to == MemoryType::MEM_L1) {
        directPath = inserter.FitL0C2L1(operation);
        return true;
    }
    bool isA5 = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510);
    if (isA5 && from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) {
        directPath = (input != nullptr) && inserter.IsL0C2UbSupportedDtype(input);
        return true;
    }
    if (isA5 && from == MemoryType::MEM_UB && to == MemoryType::MEM_L1) {
        if (MemoryPathUtils::IsMxPaddingMode(operation)) {
            directPath = false;
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Disable direct %s -> %s path for %s[%d] because MX matmul with K not 64-aligned "
                              "does not support UB2L1.",
                              BriefMemoryTypeToString(from).c_str(), BriefMemoryTypeToString(to).c_str(),
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
            return true;
        }
        directPath = inserter.FitUB2L1(operation.iOperand.front());
        return true;
    }
    return false;
}

// 特殊进阶数据通路，不满足特定条件时回退到通过DDR搬运：L0C2L1, L0C2UB, UB2L1

bool AssignMemoryType::HasParallelDifferentConsumerRequirementStandard(const LogicalTensorPtr& tensor,
                                                                       MemoryType targetType) const
{
    if (tensor == nullptr || tensor->GetConsumers().size() <= 1) {
        return false;
    }
    return HasDifferentConsumerRequirement(tensor, targetType);
}

bool AssignMemoryType::HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor,
                                                               MemoryType targetType) const
{
    return IsSliceMode() ? HasParallelDifferentConsumerRequirementStandard(tensor, targetType) :
                           HasParallelDifferentConsumerRequirementLegacy(tensor, targetType);
}

bool AssignMemoryType::HasDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const
{
    if (tensor == nullptr) {
        return false;
    }
    auto hasTerminalDifferentMoveConsumer = [targetType](const LogicalTensorPtr& branchTensor) {
        if (targetType != MemoryType::MEM_UB || branchTensor == nullptr) {
            return false;
        }
        return std::any_of(
            branchTensor->GetConsumers().begin(), branchTensor->GetConsumers().end(),
            [targetType](Operation* consumerOp) {
                if (consumerOp == nullptr ||
                    (consumerOp->GetOpcode() != Opcode::OP_CONTRACT &&
                     consumerOp->GetOpcode() != Opcode::OP_ASSEMBLE) ||
                    consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                    return false;
                }
                auto output = consumerOp->oOperand.front();
                bool needCopy = false;
                return output->GetConsumers().empty() &&
                       ((consumerOp->GetAttr<bool>("NeedCopy", needCopy) && needCopy) ||
                        MemoryPathUtils::IsDifferentKnownRequirement(output->GetMemoryTypeOriginal(), targetType));
            });
    };
    auto requirements = inserter.GetConsumerRequirements(tensor);
    bool hasDifferentRequirement = std::any_of(
        requirements.begin(), requirements.end(),
        [this, targetType, &hasTerminalDifferentMoveConsumer](const auto& item) {
            auto resolveOutputRequirement = [this](const LogicalTensorPtr& output) {
                return MemoryPathUtils::InferUniqueRequirementThroughViewConsumers(inserter, output);
            };
            Operation* consumerOp = item.first;
            if (consumerOp != nullptr &&
                (consumerOp->GetOpcode() == Opcode::OP_VIEW || consumerOp->GetOpcode() == Opcode::OP_SLICE) &&
                !consumerOp->oOperand.empty() && hasTerminalDifferentMoveConsumer(consumerOp->oOperand.front())) {
                return true;
            }
            MemoryType requirement = MemoryPathUtils::ResolveEffectiveConsumerRequirement(
                consumerOp, item.second, targetType, resolveOutputRequirement);
            if (MemoryPathUtils::IsDifferentKnownRequirement(requirement, targetType)) {
                return true;
            }
            if (consumerOp == nullptr ||
                (consumerOp->GetOpcode() != Opcode::OP_CONTRACT && consumerOp->GetOpcode() != Opcode::OP_ASSEMBLE) ||
                consumerOp->oOperand.empty() || consumerOp->oOperand.front() == nullptr) {
                return false;
            }
            auto output = consumerOp->oOperand.front();
            auto outputRequirement = output->GetMemoryTypeOriginal();
            return output->GetConsumers().empty() &&
                   MemoryPathUtils::IsDifferentKnownRequirement(outputRequirement, targetType);
        });
    if (hasDifferentRequirement) {
        return true;
    }
    return hasTerminalDifferentMoveConsumer(tensor);
}

bool AssignMemoryType::CanUseDirectViewPath(Operation& operation, MemoryType from, MemoryType to)
{
    if (from == MemoryType::MEM_UNKNOWN || to == MemoryType::MEM_UNKNOWN)
        return false;
    if (from == to)
        return true;
    if (from != MemoryType::MEM_DEVICE_DDR && to == MemoryType::MEM_DEVICE_DDR)
        return false;
    bool directPath = false;
    if (TryHandleSpecialDirectMemoryPath(operation, from, to, directPath))
        return directPath;
    std::vector<MemoryType> paths;
    bool pathFound = Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (!pathFound || paths.empty())
        return false;
    static constexpr size_t DIRECT_MEMORY_PATH_LENGTH = 2;
    bool isDirectPath = paths.size() == DIRECT_MEMORY_PATH_LENGTH && paths.front() == from && paths.back() == to;
    return isDirectPath;
}

Status AssignMemoryType::InferAssembleMemoryType(Function& function, Operation& operation,
                                                 std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_ASSEMBLE, "Infer OP_ASSEMBLE memory type", input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    if (inserter.GetRequirementOrUnknown(input, operation) != MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    if (!inferredAssembleOutputs.insert(output).second) {
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(InferAssembleMemoryType(operation));
    PropagateMemoryTypeToRawTensorSiblings(function, operation.oOperand.front(), inferredAssembleOutputs);
    return SUCCESS;
}

Status AssignMemoryType::InferAssembleMemoryType(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE)
        return SUCCESS;
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] memory type failed because operand is empty.",
                          operation.GetOpMagic());
        return FAILED;
    }
    if (std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute()) == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_ASSEMBLE[%d] memory type failed because assemble attr is null.",
                          operation.GetOpMagic());
        return FAILED;
    }
    auto output = operation.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal == MemoryType::MEM_UNKNOWN) {
        outputOriginal = InferOriginalFromOutputRequirements(output);
    }
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        RETURN_IF_NOT_SUCCESS(
            MemoryPathUtils::SetOriginalChecked(output, outputOriginal, "InferAssembleOutputRequirement"));
        return SetParallelAssembleInputRequirements(output, outputOriginal, "InferAssembleOutputRequirement");
    }
    MemoryType inputRequirement = InferParallelAssembleInputRequirement(output);
    if (inputRequirement == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    RETURN_IF_NOT_SUCCESS(
        MemoryPathUtils::SetOriginalChecked(output, inputRequirement, "InferAssembleInputRequirement"));
    return SetParallelAssembleInputRequirements(output, inputRequirement, "InferAssembleInputRequirement");
}

TensorSet AssignMemoryType::GetLogicalTensorsByRawTensor(Function& function, const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->tensor == nullptr) {
        return {};
    }
    return GraphUtils::GetTensorsByRawMagic(function, tensor->tensor->rawmagic);
}

void AssignMemoryType::PropagateMemoryTypeToRawTensorSiblings(
    Function& function, const LogicalTensorPtr& output, std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs)
{
    if (output == nullptr) {
        return;
    }
    auto siblings = GetLogicalTensorsByRawTensor(function, output);
    if (siblings.size() <= 1) {
        return;
    }
    MemoryType inferredType = output->GetMemoryTypeOriginal();
    for (const auto& sibling : siblings) {
        if (sibling == output || sibling == nullptr) {
            continue;
        }
        inferredAssembleOutputs.insert(sibling);
        if (sibling->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
            sibling->SetMemoryTypeOriginal(inferredType, false);
        }
    }
}

Status AssignMemoryType::ResolveInconsistentRawTensorMemoryTypes(Function& function)
{
    auto isMoveOutOutput = [](const LogicalTensorPtr& tensor) {
        if (tensor == nullptr) {
            return false;
        }
        return std::any_of(tensor->GetProducers().begin(), tensor->GetProducers().end(), [](const auto* producer) {
            return producer != nullptr && IsAssembleLike(producer->GetOpcode());
        });
    };
    std::unordered_set<int64_t> visitedRawMagic;
    for (auto& op : function.Operations()) {
        if (!IsAssembleLike(op.GetOpcode())) {
            continue;
        }
        if (op.oOperand.empty() || op.oOperand.front() == nullptr) {
            continue;
        }
        auto output = op.oOperand.front();
        if (output->tensor == nullptr) {
            continue;
        }
        int64_t rawMagic = output->tensor->rawmagic;
        if (!visitedRawMagic.insert(rawMagic).second) {
            continue;
        }
        auto siblings = GetLogicalTensorsByRawTensor(function, output);
        if (siblings.size() <= 1) {
            continue;
        }
        MemoryType firstType = MemoryType::MEM_UNKNOWN;
        bool hasMoveOutOutput = false;
        bool inconsistent = false;
        for (const auto& sibling : siblings) {
            if (!isMoveOutOutput(sibling)) {
                continue;
            }
            hasMoveOutOutput = true;
            MemoryType siblingType = sibling->GetMemoryTypeOriginal();
            if (siblingType == MemoryType::MEM_UNKNOWN) {
                continue;
            }
            if (firstType == MemoryType::MEM_UNKNOWN) {
                firstType = siblingType;
            } else if (siblingType != firstType) {
                inconsistent = true;
                break;
            }
        }
        if (hasMoveOutOutput && inconsistent) {
            APASS_LOG_WARN_F(Elements::Tensor,
                             "Inconsistent memory types detected on rawMagic %ld across assemble/contract outputs, "
                             "falling back to DDR.",
                             static_cast<long>(rawMagic));
            for (const auto& sibling : siblings) {
                if (isMoveOutOutput(sibling)) {
                    MemoryPathUtils::ForceSetOriginal(sibling, MemoryType::MEM_DEVICE_DDR,
                                                      "ResolveInconsistentRawTensorMemoryTypes");
                }
            }
        }
    }
    return SUCCESS;
}

MemoryType AssignMemoryType::InferParallelAssembleInputRequirement(const LogicalTensorPtr& output) const
{
    if (output == nullptr) {
        return MemoryType::MEM_UNKNOWN;
    }
    std::set<MemoryType> candidates;
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp)) {
            continue;
        }
        auto input = producerOp->iOperand.front();
        MemoryType candidate = inserter.GetRequirementOrUnknown(input, *producerOp);
        if (candidate == MemoryType::MEM_UNKNOWN) {
            candidate = InferRequirementFromInputOriginals(input);
        }
        if (candidate != MemoryType::MEM_UNKNOWN) {
            candidates.insert(candidate);
        }
    }
    if (candidates.empty()) {
        return MemoryType::MEM_UNKNOWN;
    }
    if (candidates.size() == 1) {
        return *candidates.begin();
    }
    return MemoryType::MEM_DEVICE_DDR;
}

Status AssignMemoryType::SetParallelAssembleInputRequirements(const LogicalTensorPtr& output, MemoryType memoryType,
                                                              const std::string& reason)
{
    if (output == nullptr || memoryType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    for (auto& producerOp : output->GetProducers()) {
        if (!MemoryPathUtils::IsAssembleProducer(producerOp)) {
            continue;
        }
        auto input = producerOp->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because input tensor is null.",
                              producerOp->GetOpMagic());
            return FAILED;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, *producerOp, memoryType, reason);
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producerOp->GetOpAttribute());
        if (assembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Infer OP_ASSEMBLE[%d] failed because assemble attr is null.",
                              producerOp->GetOpMagic());
            return FAILED;
        }
        assembleOpAttribute->SetFromType(memoryType);
    }
    return SUCCESS;
}

bool AssignMemoryType::CanUseDirectAssemblePath(Operation& operation, MemoryType from, MemoryType to)
{
    if (from == MemoryType::MEM_UNKNOWN || to == MemoryType::MEM_UNKNOWN) {
        return false;
    }
    if (from == to) {
        return true;
    }
    bool directPath = false;
    if (TryHandleSpecialDirectMemoryPath(operation, from, to, directPath)) {
        return directPath;
    }
    std::vector<MemoryType> paths;
    Platform::Instance().GetDie().FindNearestPath(from, to, paths);
    if (paths.empty()) {
        return false;
    }
    bool hasDdr = std::find(paths.begin(), paths.end(), MemoryType::MEM_DEVICE_DDR) != paths.end();
    return !hasDdr;
}

Status AssignMemoryType::IsAssembleToOffsetAligned(Operation& operation, const LogicalTensorPtr& output, bool& aligned)
{
    aligned = false;
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr || output == nullptr || output->GetRawTensor() == nullptr) {
        return FAILED;
    }
    const auto& rawShape = output->GetRawTensor()->rawshape;
    const auto& toOffset = assembleOpAttribute->GetToOffset();
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    static constexpr int64_t DEGENERATE_AXIS_SIZE = 1;
    static constexpr size_t MIN_RANK_WITH_SECOND_LAST_AXIS = 2;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(output->Datatype()));
    if (tensorBytes <= 0 || ASSEMBLE_ALIGN_BYTES % tensorBytes != 0 || rawShape.empty()) {
        aligned = true;
        return SUCCESS;
    }
    // Dynamic shape (-1) cannot be padded or aligned at compile time.
    if (std::find(rawShape.begin(), rawShape.end(), -1) != rawShape.end()) {
        return SUCCESS;
    }
    int64_t alignElements = ASSEMBLE_ALIGN_BYTES / tensorBytes;
    size_t lastIdx = rawShape.size() - 1;
    auto padUp = [](int64_t dim, int64_t base) { return (dim + base - 1) / base * base; };
    auto isAlignedAfterPad = [&](size_t padIdx, bool& padAligned) -> Status {
        padAligned = false;
        if (rawShape[padIdx] <= 0) {
            return SUCCESS;
        }
        Shape paddedShape = rawShape;
        paddedShape[padIdx] = padUp(rawShape[padIdx], alignElements);
        int64_t paddedOffset = 0;
        RETURN_IF_NOT_SUCCESS(CalcLineOffset(paddedShape, toOffset, paddedOffset));
        padAligned = (tensorBytes * paddedOffset) % ASSEMBLE_ALIGN_BYTES == 0;
        return SUCCESS;
    };
    // PadLocalBuffer always pads the tail axis to 32B (non axis-combine mode, or axis-combine
    // mode where tensor is not eligible). This check is mandatory.
    bool tailPadAligned = false;
    RETURN_IF_NOT_SUCCESS(isAlignedAfterPad(lastIdx, tailPadAligned));
    // When tail axis == 1 and there is a second-to-last axis, AxisCombine may pad the
    // second-to-last axis instead (ASSEMBLE is a shapeTransformOp). Since AssignMemoryType
    // cannot determine which padding will apply, both must be aligned to safely avoid DDR
    // fallback.
    if (rawShape[lastIdx] == DEGENERATE_AXIS_SIZE && rawShape.size() >= MIN_RANK_WITH_SECOND_LAST_AXIS) {
        bool secondLastPadAligned = false;
        RETURN_IF_NOT_SUCCESS(isAlignedAfterPad(lastIdx - 1, secondLastPadAligned));
        aligned = tailPadAligned && secondLastPadAligned;
        return SUCCESS;
    }
    aligned = tailPadAligned;
    return SUCCESS;
}

Status AssignMemoryType::InferReshapeMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_RESHAPE, "Infer OP_RESHAPE memory type", input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType inputOriginal = input->GetMemoryTypeOriginal();
    MemoryType inputRequirement = MemoryPathUtils::GetReshapeInputRequirement(inserter, operation, input,
                                                                              inputOriginal);
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::InferReshapeOutputFromRequirement(inserter, output, outputOriginal));
    bool kept = false;
    RETURN_IF_NOT_SUCCESS(KeepSplitReshapeUb(operation, input, output, kept));
    if (kept) {
        return SUCCESS;
    }
    bool isDynamic = MemoryPathUtils::IsDynamicReshape(operation, output);
    bool canUseUb = MemoryPathUtils::CanUseUbForReshape(input, output, inputRequirement, outputOriginal);
    return MemoryPathUtils::ApplyReshapeMemoryType(inserter, operation, input, output, isDynamic, canUseUb);
}

Status AssignMemoryType::InferViewTypeMemoryType(Operation& operation)
{
    LogicalTensorPtr input;
    LogicalTensorPtr output;
    bool shouldHandle = false;
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::GetFirstInputOutputIfOpcode(
        operation, Opcode::OP_VIEW_TYPE, "Infer OP_VIEW_TYPE memory type", input, output, shouldHandle));
    if (!shouldHandle) {
        return SUCCESS;
    }
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    MemoryType outputRequirement = output == nullptr ? MemoryType::MEM_UNKNOWN :
                                                       inserter.TryGetUniqueKnownRequiredType(output);
    // 输出 toBeMap 未知时，沿后续未推导的视图链向前查找有效内存类型
    if (outputRequirement == MemoryType::MEM_UNKNOWN) {
        MemoryType forwarded = MemoryPathUtils::InferTargetTypeThroughForwardViews(inserter, output);
        if (forwarded != MemoryType::MEM_UNKNOWN) {
            APASS_LOG_DEBUG_F(
                Elements::Operation,
                "Infer OP_VIEW_TYPE[%d] memory type reused from forward view requirement %s for output tensor[%d].",
                operation.GetOpMagic(), BriefMemoryTypeToString(forwarded).c_str(), output->GetMagic());
            outputRequirement = forwarded;
        }
    }
    MemoryType targetType = outputRequirement != MemoryType::MEM_UNKNOWN ? outputRequirement : outputOriginal;
    bool handled = false;
    RETURN_IF_NOT_SUCCESS(TryInferViewTypeFromProducerMoveIn(operation, input, output, targetType, handled));
    if (handled) {
        return SUCCESS;
    }
    return MemoryPathUtils::InferViewTypeInput(inserter, operation, input, output, targetType);
}

Status AssignMemoryType::TryInferViewTypeFromProducerMoveIn(Operation& operation, const LogicalTensorPtr& input,
                                                            const LogicalTensorPtr& output, MemoryType targetType,
                                                            bool& handled)
{
    handled = false;
    auto& producers = input->GetProducers();
    if (producers.empty()) {
        return SUCCESS;
    }
    auto producer = *producers.begin();
    if (producer == nullptr || !IsModeMoveIn(mode_, producer->GetOpcode())) {
        return SUCCESS;
    }
    handled = true;
    if (producer->iOperand.empty()) {
        const char* opcodeName = IsSliceMode() ? "OP_SLICE" : "OP_VIEW";
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Infer OP_VIEW_TYPE[%d] memory type failed because producer %s[%d] input is empty.",
                          operation.GetOpMagic(), opcodeName, producer->GetOpMagic());
        return FAILED;
    }
    auto moveInput = producer->iOperand.front();
    MemoryType moveInputRequirement = inserter.GetRequirementOrUnknown(moveInput, *producer);
    if (moveInputRequirement == MemoryType::MEM_UNKNOWN) {
        moveInputRequirement = moveInput->GetMemoryTypeOriginal();
    }
    const char* reason = IsSliceMode() ? "InferViewTypeProducerSlice" : "InferViewTypeProducerView";
    const char* fallbackReason = IsSliceMode() ? "InferViewTypeProducerSliceFallback" :
                                                 "InferViewTypeProducerViewFallback";
    if (targetType != MemoryType::MEM_UNKNOWN && CanUseDirectViewPath(*producer, moveInputRequirement, targetType)) {
        MemoryPathUtils::ForceSetOriginal(input, targetType, reason);
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, targetType, reason);
        MemoryPathUtils::ForceSetOriginal(output, targetType, reason);
        return SUCCESS;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, fallbackReason);
    MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, fallbackReason);
    return SUCCESS;
}

Status AssignMemoryType::KeepSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input,
                                            const LogicalTensorPtr& output, bool& kept)
{
    kept = false;
    if (input == nullptr || output == nullptr) {
        return SUCCESS;
    }
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    if (producers.empty() || consumers.empty()) {
        return SUCCESS;
    }
    bool allProducersContract = std::all_of(producers.begin(), producers.end(), [](const auto& producer) {
        return producer != nullptr && producer->GetOpcode() == Opcode::OP_CONTRACT;
    });
    bool allConsumersSlice = std::all_of(consumers.begin(), consumers.end(), [](const auto& consumer) {
        if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_SLICE) {
            return false;
        }
        // 后续放开在DAV_3510直接走UB2L1，暂走DDR2L1
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
        return viewOpAttribute == nullptr || viewOpAttribute->GetTo() != MemoryType::MEM_L1;
    });
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_ASSEMBLE);
    int64_t inputDataSize = input->GetDataSize();
    if (allProducersContract && allConsumersSlice && inputDataSize >= 0 &&
        static_cast<size_t>(inputDataSize) <= ubThreshold) {
        bool canKeepProducers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepContractProducersInUb(input, canKeepProducers));
        if (canKeepProducers) {
            MemoryPathUtils::ForceSetOriginal(input, MemoryType::MEM_UB, "InferSplitReshapeUb");
            MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_UB, "InferSplitReshapeUb");
            MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_UB, "InferSplitReshapeUb");
            for (const auto& consumerOp : output->GetConsumers()) {
                if (consumerOp != nullptr) {
                    MemoryPathUtils::ForceSetRequirement(inserter, output, *consumerOp, MemoryType::MEM_UB,
                                                         "InferSplitReshapeUb");
                }
            }
            kept = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::CanKeepContractProducersInUb(const LogicalTensorPtr& tensor, bool& canKeep)
{
    canKeep = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* producerOp : tensor->GetProducers()) {
        if (producerOp == nullptr || producerOp->GetOpcode() != Opcode::OP_CONTRACT || producerOp->iOperand.empty()) {
            return SUCCESS;
        }
        MemoryType fromType = MemoryPathUtils::GetAssembleInputType(inserter, *producerOp);
        constexpr MemoryType targetType = MemoryType::MEM_UB;
        bool checkOffsetAlignment = !MemoryPathUtils::IsAdvancedMemoryPath(fromType, targetType);
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(*producerOp, tensor, aligned));
        if ((checkOffsetAlignment && !aligned) || !CanUseDirectAssemblePath(*producerOp, fromType, targetType)) {
            return SUCCESS;
        }
    }
    canKeep = true;
    return SUCCESS;
}

Status AssignMemoryType::IsSliceFromOffsetAligned(Operation& sliceOp, const LogicalTensorPtr& input, bool& aligned)
{
    aligned = false;
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    if (viewOpAttribute == nullptr || input == nullptr) {
        return SUCCESS;
    }
    int64_t lineOffset = 0;
    RETURN_IF_NOT_SUCCESS(
        CalcLineOffset(input->GetRawTensor()->rawshape, viewOpAttribute->GetFromOffset(), lineOffset));
    static constexpr int ASSEMBLE_ALIGN_BYTES = 32;
    int64_t tensorBytes = static_cast<int64_t>(BytesOf(input->Datatype()));
    aligned = (tensorBytes * lineOffset) % ASSEMBLE_ALIGN_BYTES == 0;
    return SUCCESS;
}

Status AssignMemoryType::CanKeepSliceConsumersInUb(const LogicalTensorPtr& tensor, bool& canKeep)
{
    canKeep = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* consumerOp : tensor->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_SLICE || consumerOp->oOperand.empty() ||
            consumerOp->oOperand.front() == nullptr) {
            return SUCCESS;
        }
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsSliceFromOffsetAligned(*consumerOp, tensor, aligned));
        if (!aligned) {
            return SUCCESS;
        }
    }
    canKeep = true;
    return SUCCESS;
}

Status AssignMemoryType::HasNonZeroSliceFromOffset(const LogicalTensorPtr& tensor, bool& hasNonZero)
{
    hasNonZero = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    for (auto* consumerOp : tensor->GetConsumers()) {
        if (consumerOp == nullptr || consumerOp->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        if (viewAttr == nullptr) {
            continue;
        }
        int64_t lineOffset = 0;
        RETURN_IF_NOT_SUCCESS(CalcLineOffset(tensor->GetRawTensor()->rawshape, viewAttr->GetFromOffset(), lineOffset));
        if (lineOffset > 0) {
            hasNonZero = true;
            return SUCCESS;
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallback(Function& function)
{
    const size_t ubThreshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_ASSEMBLE);
    const size_t ubNormalThreshold = static_cast<size_t>(
        Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * UB_THRESHOLD_NORMAL);
    const size_t l1Threshold = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) *
                                                   L1_THRESHOLD);
    if (IsSliceMode()) {
        APASS_LOG_INFO_F(Elements::Function, "Memory threshold: UB strict %zu, UB normal %zu, L1 %zu.", ubThreshold,
                         ubNormalThreshold, l1Threshold);
    } else {
        APASS_LOG_INFO_F(Elements::Function,
                         "Memory threshold: UB assemble %zu bytes, UB normal %zu bytes, L1 %zu bytes.", ubThreshold,
                         ubNormalThreshold, l1Threshold);
    }
    for (auto& op : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(ApplyOversizedLocalBufferFallback(op));
    }
    return SUCCESS;
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallback(Operation& operation)
{
    return IsSliceMode() ? ApplyOversizedLocalBufferFallbackStandard(operation) :
                           ApplyOversizedLocalBufferFallbackLegacy(operation);
}

Status AssignMemoryType::ApplyOversizedLocalBufferFallbackStandard(Operation& operation)
{
    auto opcode = operation.GetOpcode();
    if (opcode != Opcode::OP_SLICE && opcode != Opcode::OP_CONTRACT) {
        return SUCCESS;
    }
    if (operation.iOperand.empty() || operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Apply oversized fallback for %s[%d] failed because operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    auto output = operation.oOperand.front();
    if (input == nullptr || output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because operand tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }

    if (opcode == Opcode::OP_CONTRACT) {
        MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, operation);
        if (!MemoryPathUtils::IsOversizedLocalBuffer(input, inputRequirement, true, true)) {
            return SUCCESS;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR,
                                             "ApplyOversizedContractInputFallback");
        auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
        if (assembleAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Apply oversized fallback for OP_CONTRACT[%d] failed because assemble attr is null.",
                              operation.GetOpMagic());
            return FAILED;
        }
        assembleAttr->SetFromType(MemoryType::MEM_DEVICE_DDR);
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Force OP_CONTRACT[%d] input tensor[%d] requirement to DDR by size limit.",
                          operation.GetOpMagic(), input->GetMagic());
        return SUCCESS;
    }

    if (MemoryPathUtils::IsOversizedLocalBuffer(output, output->GetMemoryTypeOriginal(), false, true)) {
        MemoryPathUtils::ForceSetOriginal(output, MemoryType::MEM_DEVICE_DDR, "ApplyOversizedSliceOutputFallback");
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
        if (viewAttr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Apply oversized fallback for OP_SLICE[%d] failed because view attr is null.",
                              operation.GetOpMagic());
            return FAILED;
        }
        viewAttr->SetToType(MemoryType::MEM_DEVICE_DDR);
        APASS_LOG_DEBUG_F(Elements::Operation, "Force OP_SLICE[%d] output tensor[%d] to DDR by size limit.",
                          operation.GetOpMagic(), output->GetMagic());
    }
    return DowngradeOversizedMoveInInputRequirement(operation);
}

Status AssignMemoryType::DowngradeOversizedMoveInInputRequirement(Operation& operation)
{
    const char* opcodeName = IsSliceMode() ? "OP_SLICE" : "OP_VIEW";
    if (operation.iOperand.empty() || operation.iOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Apply oversized fallback for %s[%d] failed because of invalid input operand.", opcodeName,
                          operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    MemoryType inputType = inserter.GetRequirementOrUnknown(input, operation);
    if (!MemoryPathUtils::IsOversizedLocalBuffer(input, inputType, false, true)) {
        return SUCCESS;
    }
    const char* reason = IsSliceMode() ? "ApplyOversizedSliceInputFallback" : "ApplyOversizedViewInputFallback";
    MemoryPathUtils::ForceSetRequirement(inserter, input, operation, MemoryType::MEM_DEVICE_DDR, reason);
    APASS_LOG_DEBUG_F(Elements::Operation, "Force %s[%d] input tensor[%d] requirement to DDR by size limit.",
                      opcodeName, operation.GetOpMagic(), input->GetMagic());
    return SUCCESS;
}

Status AssignMemoryType::ResolveMemoryUnknowns(Function& function)
{
    std::unordered_set<LogicalTensorPtr> visited;
    auto resolveTensor = [this, &function, &visited](const LogicalTensorPtr& tensor) -> Status {
        if (tensor != nullptr && !visited.insert(tensor).second) {
            return SUCCESS;
        }
        if (ShouldResolveExplicitUnknownRequirementToDdr(function, tensor)) {
            MemoryPathUtils::ForceSetOriginal(tensor, MemoryType::MEM_DEVICE_DDR,
                                              "ResolveExplicitUnknownSliceFromInCast");
            for (auto* consumer : tensor->GetConsumers()) {
                if (consumer != nullptr) {
                    MemoryPathUtils::ForceSetRequirement(inserter, tensor, *consumer, MemoryType::MEM_DEVICE_DDR,
                                                         "ResolveExplicitUnknownSliceFromInCast");
                }
            }
        }
        return MemoryPathUtils::ResolveTensorMemoryUnknowns(inserter, tensor);
    };
    for (auto& op : function.Operations()) {
        for (auto& input : op.iOperand) {
            RETURN_IF_NOT_SUCCESS(resolveTensor(input));
        }
        for (auto& output : op.oOperand) {
            RETURN_IF_NOT_SUCCESS(resolveTensor(output));
        }
    }
    return SUCCESS;
}

bool AssignMemoryType::ShouldResolveExplicitUnknownRequirementToDdr(const Function& function,
                                                                    const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetConsumers().empty()) {
        return false;
    }
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || inserter.GetRequirementOrUnknown(tensor, *consumer) != MemoryType::MEM_UNKNOWN) {
            return false;
        }
    }

    // Only apply this rule to a SLICE fed directly by an inCast.
    Operation* slice = nullptr;
    for (auto* producer : tensor->GetProducers()) {
        if (producer == nullptr || producer->GetOpcode() != Opcode::OP_SLICE || producer->oOperand.empty() ||
            producer->oOperand.front() != tensor || producer->iOperand.empty() ||
            producer->iOperand.front() == nullptr) {
            return false;
        }
        if (slice != nullptr) {
            return false;
        }
        slice = producer;
    }
    if (slice == nullptr || std::find(function.inCasts_.begin(), function.inCasts_.end(), slice->iOperand.front()) ==
                                function.inCasts_.end()) {
        return false;
    }
    if (std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute()) == nullptr) {
        return false;
    }

    // Distinguish an explicit unknown entry in opcode.cpp from an opcode with no
    // memory definition at this input position.
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr) {
            return false;
        }
        auto it = std::find(consumer->iOperand.begin(), consumer->iOperand.end(), tensor);
        if (it == consumer->iOperand.end()) {
            return false;
        }
        size_t inputIndex = static_cast<size_t>(std::distance(consumer->iOperand.begin(), it));
        const auto& definedTypes = OpcodeManager::Inst().GetInputsMemType(consumer->GetOpcode());
        if (inputIndex >= definedTypes.size() || definedTypes[inputIndex] != MemoryType::MEM_UNKNOWN) {
            return false;
        }
    }
    return true;
}

Status AssignMemoryType::SyncViewAssembleMemoryAttrs(Function& function)
{
    for (auto& operation : function.Operations()) {
        RETURN_IF_NOT_SUCCESS(SyncViewMemoryAttr(operation));
        RETURN_IF_NOT_SUCCESS(SyncAssembleMemoryAttr(operation));
    }
    return SUCCESS;
}

Status AssignMemoryType::SyncViewMemoryAttr(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_VIEW && operation.GetOpcode() != Opcode::OP_SLICE) {
        return SUCCESS;
    }
    if (operation.oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because output operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because view attr is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto output = operation.oOperand.front();
    if (output == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] toAttr failed because output tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType toType = output->GetMemoryTypeOriginal();
    if (toType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    viewOpAttribute->SetToType(toType);
    return SUCCESS;
}

Status AssignMemoryType::SyncAssembleMemoryAttr(Operation& operation)
{
    if (operation.GetOpcode() != Opcode::OP_ASSEMBLE && operation.GetOpcode() != Opcode::OP_CONTRACT) {
        return SUCCESS;
    }
    if (operation.iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because input operand is empty.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because assemble attr is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    auto input = operation.iOperand.front();
    if (input == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Sync %s[%d] fromAttr failed because input tensor is null.",
                          operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
        return FAILED;
    }
    MemoryType fromType = inserter.GetRequirementOrUnknown(input, operation);
    if (fromType == MemoryType::MEM_UNKNOWN) {
        fromType = input->GetMemoryTypeOriginal();
    }
    if (fromType == MemoryType::MEM_UNKNOWN) {
        return SUCCESS;
    }
    assembleOpAttribute->SetFromType(fromType);
    return SUCCESS;
}

Status AssignMemoryType::ResolveUnalignedUbSlices(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_SLICE || op.iOperand.empty() || op.oOperand.empty()) {
            continue;
        }
        auto input = op.iOperand.front();
        auto output = op.oOperand.front();
        if (input == nullptr || output == nullptr || input->GetMemoryTypeOriginal() != MemoryType::MEM_UB ||
            output->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            continue;
        }
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsSliceFromOffsetAligned(op, input, aligned));
        if (!aligned) {
            // A same-memory SLICE becomes a shared-buffer VIEW below. Keep unaligned UB slices
            // materialized through DDR so vector consumers receive an independently aligned buffer.
            MemoryPathUtils::ForceSetRequirement(inserter, input, op, MemoryType::MEM_DEVICE_DDR,
                                                 "ResolveUnalignedUbSlice");
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::NormalizeViewAssembleOpcodes(Function& function)
{
    if (!IsSliceMode()) {
        return SUCCESS;
    }
    for (auto& operation : function.Operations()) {
        auto opcode = operation.GetOpcode();
        bool isViewFamily = opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE;
        bool isAssembleFamily = opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT;
        if (!isViewFamily && !isAssembleFamily) {
            continue;
        }
        if (operation.iOperand.empty() || operation.oOperand.empty() || operation.iOperand.front() == nullptr ||
            operation.oOperand.front() == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation,
                              "Normalize view/assemble opcode for %s[%d] failed because operand is invalid.",
                              operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
            return FAILED;
        }
        MemoryType inputType = operation.iOperand.front()->GetMemoryTypeOriginal();
        MemoryType outputType = operation.oOperand.front()->GetMemoryTypeOriginal();
        if (inputType == MemoryType::MEM_UNKNOWN || outputType == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        Opcode normalizedOpcode;
        if (isViewFamily) {
            normalizedOpcode = inputType == outputType ? Opcode::OP_VIEW : Opcode::OP_SLICE;
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
            if (viewOpAttribute != nullptr) {
                viewOpAttribute->SetToType(outputType);
            }
        } else {
            normalizedOpcode = inputType == outputType ? Opcode::OP_ASSEMBLE : Opcode::OP_CONTRACT;
            auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
            if (assembleOpAttribute != nullptr) {
                assembleOpAttribute->SetFromType(inputType);
            }
        }
        if (opcode == normalizedOpcode) {
            continue;
        }
        std::string opcodeBeforeNormalization = operation.GetOpcodeStr();
        operation.SetOpCode(normalizedOpcode);
        APASS_LOG_DEBUG_F(Elements::Operation, "Normalize %s[%d] to %s because input is %s and output is %s.",
                          opcodeBeforeNormalization.c_str(), operation.GetOpMagic(), operation.GetOpcodeStr().c_str(),
                          BriefMemoryTypeToString(inputType).c_str(), BriefMemoryTypeToString(outputType).c_str());
    }
    return SUCCESS;
}

Status AssignMemoryType::MarkA5SimtGatherElement(Function& function)
{
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        return SUCCESS;
    }
    for (auto& op : function.Operations()) {
        if (IsGmGatherElement(op) || IsGmScatter(op)) {
            op.SetAttribute(OP_ATTR_PREFIX + "requires_simt", true);
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::InsertConvertOpsAndInferShape(Function& function)
{
    std::unordered_set<Operation*> existingOps;
    for (auto& op : function.Operations()) {
        existingOps.insert(&op);
    }
    RETURN_IF_NOT_SUCCESS(inserter.DoInsertion(function));
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    std::vector<Operation*> addedOps;
    for (auto& op : function.Operations(false)) {
        if (existingOps.find(&op) == existingOps.end()) {
            addedOps.push_back(&op);
        }
    }
    if (!addedOps.empty()) {
        if (InferShapeUtils::InferShape(function, addedOps) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "InferShape for added ops failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AssignMemoryType::PreCheck(Function& function) { return checker.DoPreCheck(function); }

Status AssignMemoryType::PostCheck(Function& function) { return checker.DoPostCheck(function); }

Status AssignMemoryType::CalcLineOffset(const Shape& shape, const Offset& offset, int64_t& lineOffset) const
{
    if (shape.size() != offset.size() || shape.empty()) {
        APASS_LOG_ERROR_F(Elements::Tensor,
                          "CalcLineOffset failed because shape size %zu != offset size %zu or shape is empty.",
                          shape.size(), offset.size());
        return FAILED;
    }
    lineOffset = 0;
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        lineOffset += offset[i - 1] * stride;
        stride *= shape[i - 1];
    }
    return SUCCESS;
}

Status AssignMemoryType::FixViewAssembleSemanticMismatch(Function& function)
{
    for (auto& op : function.Operations(false)) {
        if (op.iOperand.empty() || op.oOperand.empty()) {
            continue;
        }
        auto input = op.iOperand.front();
        auto output = op.oOperand.front();
        if (input == nullptr || output == nullptr) {
            continue;
        }
        MemoryType inputRequirement = inserter.GetRequirementOrUnknown(input, op);
        MemoryType outputOriginal = output->GetMemoryTypeOriginal();
        if (inputRequirement == MemoryType::MEM_UNKNOWN || outputOriginal == MemoryType::MEM_UNKNOWN) {
            continue;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            // VIEW 表达搬入语义；当 output==DDR 且 input requirement!=DDR 时语义违反，
            // 需要在 VIEW 之前插入 ASSEMBLE(local→DDR)。
            // 给 input 增加 DDR requirement（指向 VIEW 自身），制造 local vs DDR 冲突。
            if (outputOriginal == MemoryType::MEM_DEVICE_DDR && inputRequirement != MemoryType::MEM_DEVICE_DDR) {
                inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR, "FixViewSemanticMismatch");
                APASS_LOG_INFO_F(
                    Elements::Operation,
                    "VIEW[%d] output=DDR input requirement=%s, update input[%d] requirement to DDR to trigger "
                    "ASSEMBLE insertion.",
                    op.GetOpMagic(), BriefMemoryTypeToString(inputRequirement).c_str(), input->GetMagic());
            }
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            // ASSEMBLE 表达搬出语义；当 input requirement==DDR 且 output!=DDR 时语义违反，
            // 需要在 ASSEMBLE 之前插入 VIEW(DDR→local)。
            // 给 input 增加 output memType requirement（指向 ASSEMBLE 自身），制造 DDR vs local 冲突。
            if (inputRequirement == MemoryType::MEM_DEVICE_DDR && outputOriginal != MemoryType::MEM_DEVICE_DDR) {
                inserter.UpdateTensorTobeMap(input, op, outputOriginal, "FixAssembleSemanticMismatch");
                RETURN_IF_NOT_SUCCESS(SyncAssembleMemoryAttr(op));
                APASS_LOG_INFO_F(
                    Elements::Operation,
                    "ASSEMBLE[%d] input requirement=DDR output=%s, update input[%d] requirement to %s to trigger "
                    "VIEW insertion.",
                    op.GetOpMagic(), BriefMemoryTypeToString(outputOriginal).c_str(), input->GetMagic(),
                    BriefMemoryTypeToString(outputOriginal).c_str());
            }
        }
    }
    return SUCCESS;
}

} // namespace npu::tile_fwk
