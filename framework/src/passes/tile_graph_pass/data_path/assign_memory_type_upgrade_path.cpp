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
 * \file assign_memory_type_upgrade_path.cpp
 * \brief AssignMemoryType 平台路径升级链路（自 assign_memory_type.cpp 按主题拆出，仅移动不改逻辑）。
 */

#include "assign_memory_type.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "passes/pass_log/pass_log.h"
#include "passes/tile_graph_pass/data_path/memory_path_utils.h"

#define MODULE_NAME "AssignMemoryType"

#define RETURN_IF_NOT_SUCCESS(expr)                \
    do {                                           \
        Status assignMemoryReturnStatus = (expr);  \
        if (assignMemoryReturnStatus != SUCCESS) { \
            return assignMemoryReturnStatus;       \
        }                                          \
    } while (0)

namespace npu::tile_fwk {

namespace {
// 二维 matmul tile 用 CheckUBTileShape 分形校验 (前两维 16 对齐)；
// LiteNPU batch matmul 分形校验 (N轴对齐，L1→UB 与 L0C→UB 一致，均为 UB 目的 ND 布局的 fixpipe 写；
// LiteNPU 当前不存在 conv 场景，rank>2 且前导维全 1 的 conv tile 不可达)；
// conv 多维 tile 用末维 (W) 对齐校验，分开判断
bool CheckUBTileShapeByRank(const LogicalTensorPtr& tensor)
{
    if (tensor != nullptr && tensor->GetShape().size() > SHAPE_DIM2) {
        if (IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch()) &&
            MemoryPathUtils::AllBatchAxesAreOne(tensor->GetShape())) {
            return MemoryPathUtils::CheckUBBatchMatmulTileShapeLiteNPU(tensor);
        }
        return MemoryPathUtils::CheckUBConvTileShape(tensor);
    }
    return MemoryPathUtils::CheckUBTileShape(tensor);
}

bool IsZeroOffset(const Offset& offset, const std::vector<SymbolicScalar>& dynOffset)
{
    return TensorOffset::IsZero(offset) && std::all_of(dynOffset.begin(), dynOffset.end(), [](const auto& dim) {
               return dim.IsImmediate() && dim.Concrete() == 0;
           });
}

bool IsWholeTensor(const LogicalTensorPtr& tensor)
{
    return tensor->GetRawTensor() != nullptr && tensor->GetRawTensor()->GetRawShape() == tensor->GetShape() &&
           IsZeroOffset(tensor->GetOffset(), tensor->GetDynOffset());
}

bool IsWholeTensorContract(const Operation& contract)
{
    if (contract.iOperand.size() != 1 || contract.oOperand.size() != 1) {
        return false;
    }
    const auto& input = contract.iOperand.front();
    const auto& output = contract.oOperand.front();
    auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(contract.GetOpAttribute());
    // 单个零偏移 CONTRACT 的有效范围由输入推导，形状推导也从输入填写 fromDynValidShape。
    // 输出有效范围若大于输入，就会包含未写入的数据。合法图中两者数值一致，表达式可能多一层 Max。
    // 全量覆盖按静态形状判断。
    return attr != nullptr && IsZeroOffset(attr->GetToOffset(), attr->GetToDynOffset()) &&
           input->GetShape() == output->GetShape() && input->Datatype() == output->Datatype() &&
           input->Format() == output->Format() && IsWholeTensor(input) && IsWholeTensor(output);
}

// ND 全量元素数: 全部维度乘积
size_t ShapeNumel(const Shape& shape)
{
    size_t elems = 1;
    for (auto dim : shape) {
        elems *= static_cast<size_t>(dim);
    }
    return elems;
}

// L0C 分形 (外轴, 内轴) 按物理布局取值: 二维 matmul tile 即 (M, N); conv 多维 tile (NCHW/NCDHW)
// 对应 M = 除 C 外全部维度乘积 (N·D·H·W), N = C 维, 与 ConstrucCopyOutTile 的 L0C (M, N) 映射一致;
// 只取末两维会丢失 N/C/D 前导维, 导致容量估算严重偏小
// batch matmul UB2L1的Batch轴限制为1，取最后两根轴计算
std::pair<size_t, size_t> GetNzOuterInnerAxes(const Shape& shape, bool isConv)
{
    constexpr size_t kConvChannelIdx = 1;
    if ((shape.size() > SHAPE_DIM2 && !isConv) || shape.size() == SHAPE_DIM2) {
        return {static_cast<size_t>(shape[shape.size() - SHAPE_DIM2]), static_cast<size_t>(shape.back())};
    }
    size_t outer = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i == kConvChannelIdx) {
            continue;
        }
        outer *= static_cast<size_t>(shape[i]);
    }
    return {outer, static_cast<size_t>(shape[kConvChannelIdx])};
}
} // namespace

// 识别来自 Cube 的搬出分支，并区分全部来自矩阵宏指令的情况。
bool AssignMemoryType::CheckCubeReshapeProducers(const std::set<Operation*, LogicalTensor::CompareOp>& producers,
                                                 bool& allMacroInstrCube) const
{
    for (auto& producer : producers) {
        if (!IsModeMoveOut(mode_, producer->GetOpcode())) {
            return false;
        }
        bool hasCubeProducer = false;
        for (auto& producerIOperand : producer->iOperand) {
            for (auto& producerProducer : producerIOperand->GetProducers()) {
                hasCubeProducer = true;
                if (producerProducer->GetCoreType() != CoreType::AIC) {
                    return false;
                }
                if (!IsMatmulMacroInstrOpcode(producerProducer->GetOpcode())) {
                    allMacroInstrCube = false;
                }
            }
        }
        if (!hasCubeProducer) {
            return false;
        }
    }

    return true;
}

// 要求每条搬运分支都有下游消费者，且均运行在向量核上。
static bool HasOnlyVectorReshapeConsumers(AssignMemoryTypeMode mode,
                                          const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    for (auto& consumer : consumers) {
        if (!IsModeMoveOp(mode, consumer->GetOpcode())) {
            return false;
        }
        bool hasVectorConsumer = false;
        for (auto& consumerOOperand : consumer->oOperand) {
            for (auto& consumerConsumer : consumerOOperand->GetConsumers()) {
                hasVectorConsumer = true;
                if (consumerConsumer->GetCoreType() != CoreType::AIV) {
                    return false;
                }
            }
        }
        if (!hasVectorConsumer) {
            return false;
        }
    }

    return true;
}

// 判定 Cube 到 Vec 的 reshape 源驻留类型，每一跳可有多个并行节点，空集不匹配。
// 路径为 Cube（AIC）→ CONTRACT 搬出 → reshape → SLICE 或 CONTRACT 搬运 → Vec（AIV）。
// Kirin9050 矩阵宏指令的输出驻留 L1，普通 Cube 输出驻留 L0C，匹配失败返回 UNKNOWN。
MemoryType AssignMemoryType::InferCubeToVecSourceType(Operation& op)
{
    if (op.GetOpcode() != npu::tile_fwk::Opcode::OP_RESHAPE) {
        return MemoryType::MEM_UNKNOWN;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();

    if (producers.empty() || consumers.empty()) {
        return MemoryType::MEM_UNKNOWN;
    }

    bool allMacroInstrCube = true;
    if (!CheckCubeReshapeProducers(producers, allMacroInstrCube) || !HasOnlyVectorReshapeConsumers(mode_, consumers)) {
        return MemoryType::MEM_UNKNOWN;
    }
    return allMacroInstrCube ? MemoryType::MEM_L1 : MemoryType::MEM_L0C;
}

// ub2l1 pattern（(s)=单跳 1..N 兄弟节点）：
// producer: vector -> slice(s)/contract(s) [-> slice(s)/contract(s)] -> reshape   (1~2 跳 move)
// consumer: reshape -> slice(s)[from L1] [-> slice(s)[from L0A]] -> cube(AIC)     (宏指令无 L0A，仅 1 跳)
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

    if (!IsReshapeVecToCubeUB2L1ConsumerPattern(consumers) && !IsReshapeVecToCubeUB2L1MacroConsumerPattern(consumers)) {
        return false;
    }

    return true;
}

// ub2l1 producer 侧（reshape 上游，1~2 跳 move）：vector -> slice(s)/contract(s) [-> slice(s)/contract(s)] -> reshape
// 检查 reshape 上游是否由向量算子经一层或两层搬运供数。
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
        // 仅支持向量算子经一层或两层 SLICE、CONTRACT 搬运后接入 reshape。
        if (!((isProducerDepth1MoveOp && isProducerDepth2AllMoveOp && isProducerDepth3AllVector) ||
              (isProducerDepth1MoveOp && isProducerDepth2AllVector))) {
            return false;
        }
    }
    return true;
}

// ub2l1 consumer 侧（reshape 下游，含 L0A 中转）：reshape -> slice(s)[from L1] -> slice(s)[from L0A] -> cube(AIC)
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

// ub2l1 宏指令 consumer 侧（无 L0A 中转）：reshape -> slice(s)[from L1] -> cube(AIC macro)
bool AssignMemoryType::IsReshapeVecToCubeUB2L1MacroConsumerPattern(
    const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    for (auto& consumer : consumers) {
        bool isConsumerDepth1MoveIn = IsModeMoveIn(mode_, consumer->GetOpcode());
        bool isConsumerDepth2AllMacroCube = false;

        std::vector<bool> isConsumerDepth2MacroCube;
        for (auto& consumerOOperand : consumer->oOperand) {
            for (auto& consumerConsumer : consumerOOperand->GetConsumers()) {
                isConsumerDepth2MacroCube.push_back(consumerConsumer->GetCoreType() == CoreType::AIC &&
                                                    IsMatmulMacroInstrOpcode(consumerConsumer->GetOpcode()));
            }
        }
        isConsumerDepth2AllMacroCube = !isConsumerDepth2MacroCube.empty() &&
                                       std::all_of(isConsumerDepth2MacroCube.begin(), isConsumerDepth2MacroCube.end(),
                                                   [](bool val) { return val; });
        if (!isConsumerDepth1MoveIn || !isConsumerDepth2AllMacroCube) {
            return false;
        }
    }
    return true;
}

// 直连搬运 (L0C2UB/L12UB 的 contract，UB2L1 的 slice) 要求源与目的 tensor shape 完全一致，
// reshape 仅改变 view 语义，实际搬运两端 shape 不一致会导致精度异常，必须回退 DDR。
bool AssignMemoryType::AreAllMoveOpsInputOutputShapeIdentical(
    const std::set<Operation*, LogicalTensor::CompareOp>& moveOps)
{
    for (auto& moveOp : moveOps) {
        if (moveOp == nullptr || moveOp->iOperand.empty() || moveOp->oOperand.empty() ||
            moveOp->iOperand.front() == nullptr || moveOp->oOperand.front() == nullptr) {
            return false;
        }
        if (moveOp->iOperand.front()->GetShape() != moveOp->oOperand.front()->GetShape()) {
            return false;
        }
    }
    return true;
}

// 将 Cube 的输出直接搬入 UB，并同步 reshape 与下游向量分支的需求。
void AssignMemoryType::ApplyReshapeCubeToVecUpgrade(Operation& op, const LogicalTensorPtr& output,
                                                    const std::set<Operation*, LogicalTensor::CompareOp>& producers,
                                                    const std::set<Operation*, LogicalTensor::CompareOp>& consumers,
                                                    MemoryType cubeOutputType)
{
    for (auto& producer : producers) {
        auto& producerInput = producer->iOperand.front();
        auto& producerOutput = producer->oOperand.front();

        // 将生产者的 CONTRACT 输入设为 Cube 输出类型（L0C 或 L1）。
        if (producerInput->GetMemoryTypeOriginal() != cubeOutputType) {
            APASS_LOG_DEBUG_F(
                Elements::Tensor, "AssignMemoryType::ApplyReshapeCubeToVecUpgrade: tensor[%d] original %s -> %s.",
                producerInput->GetMagic(), BriefMemoryTypeToString(producerInput->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(cubeOutputType).c_str());
        }
        producerInput->SetMemoryTypeOriginal(cubeOutputType, true);
        inserter.UpdateTensorTobeMap(producerInput, *producer, cubeOutputType);

        // 将生产者输出驻留在 UB。
        if (producerOutput->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            APASS_LOG_DEBUG_F(
                Elements::Tensor, "AssignMemoryType::ApplyReshapeCubeToVecUpgrade: tensor[%d] original %s -> %s.",
                producerOutput->GetMagic(), BriefMemoryTypeToString(producerOutput->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(MemoryType::MEM_UB).c_str());
        }
        producerOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        inserter.UpdateTensorTobeMap(producerOutput, op, MemoryType::MEM_UB);
    }
    // 将 reshape 输出驻留在 UB。
    if (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "AssignMemoryType::ApplyReshapeCubeToVecUpgrade: tensor[%d] original %s -> %s.",
                          output->GetMagic(), BriefMemoryTypeToString(output->GetMemoryTypeOriginal()).c_str(),
                          BriefMemoryTypeToString(MemoryType::MEM_UB).c_str());
    }
    output->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

    // 将下游 SLICE、CONTRACT 的输入需求均设为 UB。
    for (auto& consumer : consumers) {
        inserter.UpdateTensorTobeMap(output, *consumer, MemoryType::MEM_UB);
    }
}

// 将 reshape 下游搬运留在 UB 到 L1 路径，并同步后继消费者需求。
void AssignMemoryType::SetVecToCubeReshapeConsumersToL1(const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    for (auto& consumer : consumers) {
        auto& consumerInput = consumer->iOperand.front();
        auto& consumerOutput = consumer->oOperand.front();

        // 将 reshape 输出驻留在 UB，并将消费需求设为 L1。
        if (consumerInput->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            APASS_LOG_DEBUG_F(
                Elements::Tensor, "AssignMemoryType::ApplyReshapeVecToCubeUpgrade: tensor[%d] original %s -> %s.",
                consumerInput->GetMagic(), BriefMemoryTypeToString(consumerInput->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(MemoryType::MEM_UB).c_str());
        }
        consumerInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        inserter.UpdateTensorTobeMap(consumerInput, *consumer, MemoryType::MEM_L1);

        // 将消费侧切片的输出驻留在 L1。
        if (consumerOutput->GetMemoryTypeOriginal() != MemoryType::MEM_L1) {
            APASS_LOG_DEBUG_F(
                Elements::Tensor, "AssignMemoryType::ApplyReshapeVecToCubeUpgrade: tensor[%d] original %s -> %s.",
                consumerOutput->GetMagic(), BriefMemoryTypeToString(consumerOutput->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(MemoryType::MEM_L1).c_str());
        }
        consumerOutput->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
        for (auto& consumerConsumer : consumerOutput->GetConsumers()) {
            inserter.UpdateTensorTobeMap(consumerOutput, *consumerConsumer, MemoryType::MEM_L1);
        }
    }
}

// 将 Vec 到 Cube 的 reshape 输入留在 UB，并设置下游搬运到 L1。
void AssignMemoryType::ApplyReshapeVecToCubeUpgrade(Operation& op,
                                                    const std::set<Operation*, LogicalTensor::CompareOp>& producers,
                                                    const std::set<Operation*, LogicalTensor::CompareOp>& consumers)
{
    for (auto& producer : producers) {
        auto& producerInput = producer->iOperand.front();
        auto& producerOutput = producer->oOperand.front();

        // 将生产侧 SLICE、CONTRACT 的输入设为 UB。
        if (producerInput->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            APASS_LOG_DEBUG_F(
                Elements::Tensor, "AssignMemoryType::ApplyReshapeVecToCubeUpgrade: tensor[%d] original %s -> %s.",
                producerInput->GetMagic(), BriefMemoryTypeToString(producerInput->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(MemoryType::MEM_UB).c_str());
        }
        producerInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        inserter.UpdateTensorTobeMap(producerInput, *producer, MemoryType::MEM_UB);

        // 将 reshape 的输入设为 UB。
        if (producerOutput->GetMemoryTypeOriginal() != MemoryType::MEM_UB) {
            APASS_LOG_DEBUG_F(
                Elements::Tensor, "AssignMemoryType::ApplyReshapeVecToCubeUpgrade: tensor[%d] original %s -> %s.",
                producerOutput->GetMagic(), BriefMemoryTypeToString(producerOutput->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(MemoryType::MEM_UB).c_str());
        }
        producerOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        inserter.UpdateTensorTobeMap(producerOutput, op, MemoryType::MEM_UB);
    }

    SetVecToCubeReshapeConsumersToL1(consumers);
}

// 仅在 UB 容量、搬运形状和数据类型满足约束时升级 Cube 到 Vec 的 reshape。
bool AssignMemoryType::TryUpgradeCubeToVecReshape(Operation& op, const LogicalTensorPtr& input,
                                                  const LogicalTensorPtr& output)
{
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();
    // l0c2ub / l12ub pattern (batchmatmul case)，结构见 InferCubeToVecSourceType 注释。
    // 仅按 cube 输出驻留类型区分：宏指令 (kirin9050, 无 L0C) 为 L1 (l12ub)，
    // 普通 lite NPU 为 L0C (l0c2ub)，producer contract 输入随之设为 L1/L0C。
    const MemoryType cubeOutputType = InferCubeToVecSourceType(op);
    if (cubeOutputType != MemoryType::MEM_UNKNOWN && MemoryPathUtils::FitsTensorInUb(input) &&
        AreAllMoveOpsInputOutputShapeIdentical(producers) &&
        (cubeOutputType == MemoryType::MEM_L1 ? inserter.IsL12UbLiteNPUSupportedDtype(input) :
                                                inserter.IsL0C2UbLiteNPUSupportedDtype(input))) {
        ApplyReshapeCubeToVecUpgrade(op, output, producers, consumers, cubeOutputType);
        return true;
    }

    if (cubeOutputType != MemoryType::MEM_UNKNOWN) {
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Reject cube-to-vector reshape[%d] upgrade: UB capacity, transfer shape or dtype constraints; "
            "input tensor[%d] shape=%s, dtype=%s.",
            op.GetOpMagic(), input->GetMagic(), IntVecToStr(input->GetShape()).c_str(),
            DataType2String(input->Datatype()));
    }

    return false;
}

// 在 LiteNPU 上优先匹配 Cube 到 Vec，未升级时再匹配 Vec 到 Cube 的直连路径。
Status AssignMemoryType::InferReshapeL0C2UBAndUB2L1PatternLiteNPU(Operation& op)
{
    if (!IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        return SUCCESS;
    }

    auto& input = op.iOperand.front();
    auto& output = op.oOperand.front();
    auto& producers = input->GetProducers();
    auto& consumers = output->GetConsumers();

    auto hasOutcastOperand = [this](const Operation* move) {
        return move != nullptr && ((!move->iOperand.empty() && IsOutCastTensor(move->iOperand.front())) ||
                                   (!move->oOperand.empty() && IsOutCastTensor(move->oOperand.front())));
    };
    if (IsSliceMode() && (IsOutCastTensor(input) || IsOutCastTensor(output) ||
                          std::any_of(producers.begin(), producers.end(), hasOutcastOperand))) {
        return SUCCESS;
    }

    if (TryUpgradeCubeToVecReshape(op, input, output)) {
        return SUCCESS;
    }

    // ub2l1 pattern，结构见 IsReshapeVecToCubeUB2L1Pattern 注释。
    // MXMatmul 场景 K 轴非 64 对齐不支持 UB2L1 直连（MX 补齐仅由 DDR 路径支持），回退 DDR。
    if (IsReshapeVecToCubeUB2L1Pattern(op) && MemoryPathUtils::FitsTensorInUb(output) &&
        AreAllMoveOpsInputOutputShapeIdentical(consumers) && inserter.IsUb2L1LiteNPUSupportedDtype(output) &&
        !MemoryPathUtils::HasMxPaddingModeConsumer(output) &&
        (!IsSliceMode() || std::none_of(consumers.begin(), consumers.end(), hasOutcastOperand))) {
        ApplyReshapeVecToCubeUpgrade(op, producers, consumers);
        return SUCCESS;
    }

    return SUCCESS;
}

// 按平台能力及固定顺序尝试路径升级，前序规则写入的类型供后续规则使用。
Status AssignMemoryType::ApplyPlatformPathUpgradeRules(Function& function)
{
    // 按原图顺序收集候选算子。升级期间算子和 opcode 保持稳定，各规则读取当前图状态。
    std::vector<Operation*> slices;
    std::vector<Operation*> contracts;
    for (auto& op : function.Operations()) {
        const auto opcode = op.GetOpcode();
        if (opcode == Opcode::OP_SLICE) {
            slices.push_back(&op);
        } else if (opcode == Opcode::OP_CONTRACT) {
            contracts.push_back(&op);
        }
    }

    RETURN_IF_NOT_SUCCESS(ProcessDdrMultiReshape(contracts));
    RETURN_IF_NOT_SUCCESS(ProcessL1DdrL1(contracts));
    RETURN_IF_NOT_SUCCESS(ProcessL0C2L1LargeToSmall(contracts));
    RETURN_IF_NOT_SUCCESS(ProcessL0C2L1SmallToLarge(slices));
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 ||
        IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch())) {
        RETURN_IF_NOT_SUCCESS(ProcessL0C2UBEqualShape(contracts));
        RETURN_IF_NOT_SUCCESS(ProcessL0C2UBSmallToLarge(slices));
        RETURN_IF_NOT_SUCCESS(ProcessL0C2UBLargeToSmall(contracts));
        RETURN_IF_NOT_SUCCESS(ProcessUB2L1SmallToLarge(slices));
        RETURN_IF_NOT_SUCCESS(ProcessUB2L1LargeToSmall(contracts));
    }
    if (Platform::Instance().GetSoc().IsSupportCubeEpilogue()) {
        RETURN_IF_NOT_SUCCESS(ProcessL12UBSmallToLarge(slices));
        RETURN_IF_NOT_SUCCESS(ProcessL12UBLargeToSmall(contracts));
    }
    RETURN_IF_NOT_SUCCESS(ProcessUB2UBContractSlice(contracts));
    return SUCCESS;
}

static bool IsDdrReshapeUbSlice(const Operation& slice, const LogicalTensorPtr& input)
{
    return slice.GetOpcode() == Opcode::OP_SLICE && slice.iOperand.size() == 1 && slice.oOperand.size() == 1 &&
           slice.iOperand.front() == input && slice.oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
           std::dynamic_pointer_cast<ViewOpAttribute>(slice.GetOpAttribute()) != nullptr;
}

// 所有分支匹配后，依次更新主链、VIEW、分支 reshape 和切片的存储类型。
Status AssignMemoryType::ApplyDdrMultiReshapeUpgrade(Operation& contract, Operation& firstReshape,
                                                     const DdrReshapeBranches& branches)
{
    constexpr MemoryType kUbMemoryType = MemoryType::MEM_UB;
    constexpr const char* kReason = "ProcessDdrMultiReshape";
    const auto contractInput = contract.iOperand.front();
    const auto contractOutput = contract.oOperand.front();
    const auto firstReshapeOutput = firstReshape.oOperand.front();
    auto contractAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(contract.GetOpAttribute());
    if (contractAttr == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessDdrMultiReshape failed because contract attr is null.");
        return FAILED;
    }
    MemoryPathUtils::ForceSetRequirement(inserter, contractInput, contract, kUbMemoryType, kReason);
    contractAttr->SetFromType(kUbMemoryType);
    MemoryPathUtils::ForceSetOriginal(contractOutput, kUbMemoryType, kReason);
    MemoryPathUtils::ForceSetRequirement(inserter, contractOutput, firstReshape, kUbMemoryType, kReason);
    MemoryPathUtils::ForceSetOriginal(firstReshapeOutput, kUbMemoryType, kReason);

    for (auto* view : branches.branchViews) {
        auto viewOutput = view->oOperand.front();
        MemoryPathUtils::ForceSetRequirement(inserter, firstReshapeOutput, *view, kUbMemoryType, kReason);
        MemoryPathUtils::ForceSetOriginal(viewOutput, kUbMemoryType, kReason);
        std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute())->SetToType(kUbMemoryType);
    }
    for (auto* reshape : branches.branchReshapes) {
        auto reshapeInput = reshape->iOperand.front();
        MemoryPathUtils::ForceSetRequirement(inserter, reshapeInput, *reshape, kUbMemoryType, kReason);
        MemoryPathUtils::ForceSetOriginal(reshape->oOperand.front(), kUbMemoryType, kReason);
    }
    for (auto* slice : branches.branchSlices) {
        auto sliceInput = slice->iOperand.front();
        MemoryPathUtils::ForceSetRequirement(inserter, sliceInput, *slice, kUbMemoryType, kReason);
        MemoryPathUtils::ForceSetOriginal(slice->oOperand.front(), kUbMemoryType, kReason);
        std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute())->SetToType(kUbMemoryType);
    }
    for (auto* slice : branches.directSlices) {
        MemoryPathUtils::ForceSetRequirement(inserter, firstReshapeOutput, *slice, kUbMemoryType, kReason);
        MemoryPathUtils::ForceSetOriginal(slice->oOperand.front(), kUbMemoryType, kReason);
        std::dynamic_pointer_cast<ViewOpAttribute>(slice->GetOpAttribute())->SetToType(kUbMemoryType);
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Upgrade contract[%d] reshape branches from DDR to UB for UB slice outputs.",
                      contract.GetOpMagic());
    return SUCCESS;
}

// 匹配 VIEW → reshape → SLICE 分支，要求每条末端切片写入 UB。
bool AssignMemoryType::CollectDdrReshapeViewBranch(Operation& branch, const LogicalTensorPtr& input,
                                                   DdrReshapeBranches& branches) const
{
    if (branch.GetOpcode() != Opcode::OP_VIEW || branch.iOperand.size() != 1 || branch.oOperand.size() != 1 ||
        branch.iOperand.front() != input ||
        std::dynamic_pointer_cast<ViewOpAttribute>(branch.GetOpAttribute()) == nullptr) {
        return false;
    }
    auto viewOutput = branch.oOperand.front();
    if (viewOutput->GetConsumers().empty() || IsOutCastTensor(viewOutput)) {
        return false;
    }
    branches.branchViews.push_back(&branch);
    for (auto* reshape : viewOutput->GetConsumers()) {
        if (reshape->GetOpcode() != Opcode::OP_RESHAPE || reshape->iOperand.size() != 1 ||
            reshape->oOperand.size() != 1 || reshape->iOperand.front() != viewOutput) {
            return false;
        }
        auto reshapeOutput = reshape->oOperand.front();
        if (reshapeOutput->GetConsumers().empty() || IsOutCastTensor(reshapeOutput)) {
            return false;
        }
        branches.branchReshapes.push_back(reshape);
        for (auto* slice : reshapeOutput->GetConsumers()) {
            if (!IsDdrReshapeUbSlice(*slice, reshapeOutput)) {
                return false;
            }
            branches.branchSlices.push_back(slice);
        }
    }
    return true;
}

// 同时匹配直接切片和视图分支，至少有一条分支包含额外的 reshape 才参与升级。
bool AssignMemoryType::CollectDdrReshapeBranches(const LogicalTensorPtr& input, DdrReshapeBranches& branches) const
{
    for (auto* branch : input->GetConsumers()) {
        if (branch->GetOpcode() == Opcode::OP_SLICE) {
            if (!IsDdrReshapeUbSlice(*branch, input)) {
                return false;
            }
            branches.directSlices.push_back(branch);
            continue;
        }
        if (!CollectDdrReshapeViewBranch(*branch, input, branches)) {
            return false;
        }
    }
    return !branches.branchReshapes.empty();
}

// 对齐的 CONTRACT 后若所有 reshape 分支均落到 UB，将中间 DDR 驻留升级到 UB。
Status AssignMemoryType::ProcessDdrMultiReshape(const std::vector<Operation*>& contracts)
{
    constexpr MemoryType kUbMemoryType = MemoryType::MEM_UB;
    for (auto* contractOp : contracts) {
        auto& contract = *contractOp;
        auto contractInput = contract.iOperand.front();
        auto contractOutput = contract.oOperand.front();
        if (contractOutput->GetProducers().size() != 1 || contractInput->GetMemoryTypeOriginal() != kUbMemoryType ||
            contractOutput->GetConsumers().size() != 1 || IsOutCastTensor(contractOutput)) {
            continue;
        }
        bool aligned = false;
        RETURN_IF_NOT_SUCCESS(IsAssembleToOffsetAligned(contract, contractOutput, aligned));
        if (!aligned) {
            continue;
        }
        auto* firstReshape = *contractOutput->GetConsumers().begin();
        if (firstReshape->GetOpcode() != Opcode::OP_RESHAPE || firstReshape->iOperand.size() != 1 ||
            firstReshape->oOperand.size() != 1 || firstReshape->iOperand.front() != contractOutput) {
            continue;
        }
        auto firstReshapeOutput = firstReshape->oOperand.front();
        if (firstReshapeOutput->GetConsumers().empty() || IsOutCastTensor(firstReshapeOutput)) {
            continue;
        }

        DdrReshapeBranches branches;
        if (!CollectDdrReshapeBranches(firstReshapeOutput, branches)) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(ApplyDdrMultiReshapeUpgrade(contract, *firstReshape, branches));
    }
    return SUCCESS;
}

// 优先检查 CONTRACT 的输入需求，未知时才依据输入驻留类型判定是否来自 L1。
bool AssignMemoryType::AreAllContractInputsFromL1(const LogicalTensorPtr& middle) const
{
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr) {
            return false;
        }
        MemoryType requirement = inserter.GetRequirementOrUnknown(input, *producer);
        if (requirement != MemoryType::MEM_L1 &&
            (requirement != MemoryType::MEM_UNKNOWN || input->GetMemoryTypeOriginal() != MemoryType::MEM_L1)) {
            return false;
        }
    }
    return true;
}

// 当 CONTRACT 与 SLICE 两侧均可留在 L1 时，消除中间 DDR 驻留。
Status AssignMemoryType::ProcessL1DdrL1(const std::vector<Operation*>& contracts)
{
    std::unordered_set<LogicalTensorPtr> visitedOutputs;
    for (auto* contractOp : contracts) {
        auto& contract = *contractOp;
        if (contract.oOperand.empty()) {
            continue;
        }
        auto middle = contract.oOperand.front();
        if (middle == nullptr || IsOutCastTensor(middle) || !visitedOutputs.insert(middle).second ||
            (middle->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN &&
             middle->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) ||
            MemoryPathUtils::IsOversizedLocalBuffer(middle, MemoryType::MEM_L1, false, true) ||
            !HasOnlyContractProducers(middle) || !HasOnlySliceConsumers(middle)) {
            continue;
        }

        if (!AreAllContractInputsFromL1(middle)) {
            continue;
        }

        bool allSliceConsumersToL1 = true;
        for (auto* consumer : middle->GetConsumers()) {
            auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute());
            if (viewAttr == nullptr || viewAttr->GetTo() != MemoryType::MEM_L1 ||
                inserter.GetRequirementOrUnknown(middle, *consumer) != MemoryType::MEM_DEVICE_DDR) {
                allSliceConsumersToL1 = false;
                break;
            }
        }
        if (!allSliceConsumersToL1) {
            continue;
        }

        MemoryPathUtils::ForceSetOriginal(middle, MemoryType::MEM_L1, "ProcessL1DdrL1");
        for (auto* consumer : middle->GetConsumers()) {
            MemoryPathUtils::ForceSetRequirement(inserter, middle, *consumer, MemoryType::MEM_L1, "ProcessL1DdrL1");
        }
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Upgrade tensor[%d] from L1 -> DDR -> L1 path to L1 for contract/slice layout ops.",
                          middle->GetMagic());
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2L1SmallToLarge(const std::vector<Operation*>& slices)
{
    for (auto* sliceOp : slices) {
        auto& op = *sliceOp;
        RETURN_IF_NOT_SUCCESS(TryUpgradeSliceContractPath(op, MemoryType::MEM_L0C, MemoryType::MEM_L1,
                                                          "ProcessL0C2L1SmallToLarge", false, false, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessUB2UBContractSlice(const std::vector<Operation*>& contracts)
{
    for (auto* contractOp : contracts) {
        auto& op = *contractOp;
        if (op.oOperand.empty() || op.oOperand.front() == nullptr) {
            continue;
        }
        auto middle = op.oOperand.front();
        if (!MemoryPathUtils::FitsAssembleOutputMemoryLimit(middle, MemoryType::MEM_UB)) {
            continue;
        }
        bool canKeepProducers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepContractProducersInUb(middle, canKeepProducers));
        if (!canKeepProducers) {
            continue;
        }
        bool canKeepConsumers = false;
        RETURN_IF_NOT_SUCCESS(CanKeepSliceConsumersInUb(middle, canKeepConsumers));
        if (!canKeepConsumers) {
            continue;
        }
        bool hasNonZero = false;
        RETURN_IF_NOT_SUCCESS(HasNonZeroSliceFromOffset(middle, hasNonZero));
        if (hasNonZero) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                                                "ProcessUB2UBContractSlice", false, false, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2L1LargeToSmall(const std::vector<Operation*>& contracts)
{
    for (auto* contractOp : contracts) {
        auto& op = *contractOp;
        if (op.oOperand.empty() || op.oOperand.front() == nullptr) {
            continue;
        }
        // 由 TryUpgradeSingleContractSlicePath 检查单 CONTRACT 约束。
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_L0C, MemoryType::MEM_L1,
                                                                "ProcessL0C2L1LargeToSmall", false, false, false));
    }
    return SUCCESS;
}

bool AssignMemoryType::CheckConsumerMoveInShapeMultiple(const LogicalTensorPtr& output, const LogicalTensorPtr& input)
{
    for (auto& consumerOp : output->GetConsumers()) {
        if (!IsModeMoveIn(mode_, consumerOp->GetOpcode())) {
            continue;
        }
        if (!IsSliceMode()) {
            // VIEW 是单输入、单输出算子。
            ASSERT(consumerOp->GetOOperands().size() == 1) << "VIEW should have 1 output";
        }
        if (!MemoryPathUtils::IsDimMultiple(consumerOp->GetOOperands().front()->GetShape(), input->GetShape())) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::AreAllSliceConsumerShapesPreserved(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetConsumers().empty()) {
        return false;
    }
    const auto& inputShape = tensor->GetShape();
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || consumer->GetOpcode() != Opcode::OP_SLICE) {
            return false;
        }
        if (consumer->iOperand.empty() || consumer->iOperand.front() != tensor) {
            return false;
        }
        if (consumer->oOperand.empty() || consumer->oOperand.front() == nullptr ||
            consumer->oOperand.front()->GetShape() != inputShape) {
            return false;
        }
    }
    return true;
}

// 偏移是否携带非零分量: 逐元素判定, 不经 CalcLineOffset 折算行偏移。
// rawshape 含动态维(-1)时折算结果不可靠(负 stride 会把非零偏移算成 <= 0 而漏判),
// 而"是否携带偏移"只需看分量本身, 与 shape 无关。
bool AssignMemoryType::HasNonZeroOffsetElement(const Offset& offset) const
{
    return std::any_of(offset.begin(), offset.end(), [](int64_t value) { return value != 0; });
}

// 目的侧直连消费检查: 从 tensor 沿消费方向仅穿透寄存器拷贝(register_copy)链, 链上存在
// ASSEMBLE/CONTRACT 消费者即判定目的侧存在直连消费(不看 toOffset 值——FoldL0C2UBCopyOffset
// 的失败条件只看 copy 的 fromOffset 是否非立即零, assemble 的 toOffset 是否为零不影响其
// abort 与否, 故 to=0 的直连 assemble 同样必须拦截); view/slice 及计算算子为链路终点,
// 不穿透——中间隔着这些算子时已不构成同一条直连通路的消费关系。
Status AssignMemoryType::HasDirectAssembleConsumerDownstream(const LogicalTensorPtr& tensor, bool& hasDirect)
{
    hasDirect = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    std::vector<LogicalTensorPtr> pending = {tensor};
    std::unordered_set<LogicalTensorPtr> visited;
    while (!pending.empty()) {
        auto current = pending.back();
        pending.pop_back();
        if (current == nullptr || !visited.insert(current).second) {
            continue;
        }
        for (auto* consumerOp : current->GetConsumers()) {
            if (consumerOp == nullptr) {
                continue;
            }
            if (consumerOp->GetOpcode() == Opcode::OP_REGISTER_COPY) {
                for (auto& nextTensor : consumerOp->GetOOperands()) {
                    pending.push_back(nextTensor);
                }
                continue;
            }
            if (consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE || consumerOp->GetOpcode() == Opcode::OP_CONTRACT) {
                hasDirect = true;
                return SUCCESS;
            }
        }
    }
    return SUCCESS;
}

// 源侧偏移收集: 从 tensor 沿生产方向仅穿透寄存器拷贝(register_copy)链, 遇到携带非零
// fromOffset 的 SLICE/VIEW 即判定源侧存在偏移(直连 copy 将以该偏移读取源 buffer 内部
// tile); 其余算子均为链路终点, 不穿透。偏移携带口径与 fold 对齐: 静态分量逐元素非零,
// 或符号偏移(fromDynOffset)非空, 均视为携带——fold 侧判定为"非立即零即携带"。
Status AssignMemoryType::HasNonZeroViewFromOffsetUpstream(const LogicalTensorPtr& tensor, bool& hasNonZero)
{
    hasNonZero = false;
    if (tensor == nullptr) {
        return SUCCESS;
    }
    std::vector<LogicalTensorPtr> pending = {tensor};
    std::unordered_set<LogicalTensorPtr> visited;
    while (!pending.empty()) {
        auto current = pending.back();
        pending.pop_back();
        if (current == nullptr || !visited.insert(current).second) {
            continue;
        }
        for (auto* producerOp : current->GetProducers()) {
            if (producerOp == nullptr) {
                continue;
            }
            if (producerOp->GetOpcode() == Opcode::OP_REGISTER_COPY) {
                for (auto& nextTensor : producerOp->GetIOperands()) {
                    pending.push_back(nextTensor);
                }
                continue;
            }
            if (producerOp->GetOpcode() == Opcode::OP_SLICE || producerOp->GetOpcode() == Opcode::OP_VIEW) {
                auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(producerOp->GetOpAttribute());
                if (viewAttr != nullptr &&
                    (HasNonZeroOffsetElement(viewAttr->GetFromOffset()) || !viewAttr->GetFromDynOffset().empty())) {
                    hasNonZero = true;
                    return SUCCESS;
                }
            }
        }
    }
    return SUCCESS;
}

// 跨内存直连升级的双侧偏移校验(通用, 与具体内存类型无关):
// 直连升级会把 contract/slice 边界物化为一条直连 copy —— 源侧(sourceType)读取 + 目的侧
// (targetType)写入。不变量: 同一条直连通路上, 源侧累计读取偏移与目的侧累计写入偏移不得
// 同时非零——双侧偏移无法被底层直写正确表达, 也无法被 ReplaceTensor 的直写折叠(如
// FoldL0C2UBCopyOffset)消化。偏移携带口径与 fold 对齐: 静态分量逐元素非零, 或符号偏移
// (dynOffset)非空, 均视为携带。contract/slice 图中直连跨越只有两种形态, 按入口 op 的
// opcode 区分:
//   SLICE 入口(SmallToLarge, middle=slice 输入升级后驻留 targetType): 跨越点为生产者
//     contract —— dst = contract 自身 toOffset(写目的侧 middle),
//     from = contract 输入沿生产方向仅穿透 register_copy 链遇到的 view/slice fromOffset;
//   CONTRACT 入口(LargeToSmall, middle=contract 输出升级后驻留 sourceType): 跨越点为
//     slice 消费者 —— from = slice 自身 fromOffset(源侧 middle 的读取位置),
//     dst = slice 输出沿消费方向仅穿透 register_copy 链存在 assemble/contract 直连消费
//     (不看 toOffset 值——fold 的失败条件只看 copy 的 fromOffset 是否非立即零, assemble
//     的 toOffset 为零不影响其 abort, 故 from≠0 + 直连 assemble(to=0) 同样必须拦截;
//     copy 输出直接喂 vec 计算时 fold 前置条件不成立, 不受此条约束)。
// SLICE 的上游搬出若同时携带源和目的偏移，则保留原路径。
Status AssignMemoryType::ShouldSkipSliceUpgradeForOffsets(Operation& op, MemoryType sourceType, MemoryType targetType,
                                                          const std::string& reason, bool& skip)
{
    if (op.iOperand.empty() || op.iOperand.front() == nullptr) {
        return SUCCESS;
    }
    auto middle = op.iOperand.front();
    bool toNonZero = false;
    for (auto* producer : middle->GetProducers()) {
        if (producer == nullptr) {
            continue;
        }
        auto assembleAttr = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (assembleAttr != nullptr &&
            (HasNonZeroOffsetElement(assembleAttr->GetToOffset()) || !assembleAttr->GetToDynOffset().empty())) {
            toNonZero = true;
            break;
        }
    }
    if (!toNonZero) {
        return SUCCESS;
    }
    bool fromNonZero = false;
    for (auto* producer : middle->GetProducers()) {
        if (producer == nullptr || producer->GetIOperands().empty()) {
            continue;
        }
        bool producerFromNonZero = false;
        RETURN_IF_NOT_SUCCESS(HasNonZeroViewFromOffsetUpstream(producer->GetIOperands().front(), producerFromNonZero));
        if (producerFromNonZero) {
            fromNonZero = true;
            break;
        }
    }
    if (fromNonZero) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Skip %s upgrade for tensor[%d]: %s->%s direct path carries both from and to offsets.",
                          reason.c_str(), middle->GetMagic(), BriefMemoryTypeToString(sourceType).c_str(),
                          BriefMemoryTypeToString(targetType).c_str());
        skip = true;
    }
    return SUCCESS;
}

// 若 CONTRACT 的下游切片带源偏移且仍有直接拼接消费者，则保留原路径。
Status AssignMemoryType::ShouldSkipContractUpgradeForOffsets(Operation& op, MemoryType sourceType,
                                                             MemoryType targetType, const std::string& reason,
                                                             bool& skip)
{
    if (op.oOperand.empty() || op.oOperand.front() == nullptr) {
        return SUCCESS;
    }
    auto middle = op.oOperand.front();
    bool fromNonZero = false;
    for (auto* sliceConsumer : middle->GetConsumers()) {
        if (sliceConsumer == nullptr || sliceConsumer->GetOpcode() != Opcode::OP_SLICE) {
            continue;
        }
        auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(sliceConsumer->GetOpAttribute());
        if (viewAttr != nullptr &&
            (HasNonZeroOffsetElement(viewAttr->GetFromOffset()) || !viewAttr->GetFromDynOffset().empty())) {
            fromNonZero = true;
            break;
        }
    }
    if (!fromNonZero) {
        return SUCCESS;
    }
    bool dstPresent = false;
    for (auto* sliceConsumer : middle->GetConsumers()) {
        if (sliceConsumer == nullptr || sliceConsumer->GetOpcode() != Opcode::OP_SLICE ||
            sliceConsumer->GetOOperands().empty()) {
            continue;
        }
        bool sliceDstPresent = false;
        RETURN_IF_NOT_SUCCESS(
            HasDirectAssembleConsumerDownstream(sliceConsumer->GetOOperands().front(), sliceDstPresent));
        if (sliceDstPresent) {
            dstPresent = true;
            break;
        }
    }
    if (dstPresent) {
        APASS_LOG_DEBUG_F(Elements::Tensor,
                          "Skip %s upgrade for tensor[%d]: %s->%s direct path carries non-zero from offset "
                          "with direct assemble consumer.",
                          reason.c_str(), middle->GetMagic(), BriefMemoryTypeToString(sourceType).c_str(),
                          BriefMemoryTypeToString(targetType).c_str());
        skip = true;
    }
    return SUCCESS;
}

// 分别检查 SLICE 与 CONTRACT 两侧的偏移和下游写入，保留无法安全直连的搬运边界。
Status AssignMemoryType::ShouldSkipDirectPathUpgradeForBothSideOffsets(Operation& op, MemoryType sourceType,
                                                                       MemoryType targetType, const std::string& reason,
                                                                       bool& skip)
{
    skip = false;
    if (op.GetOpcode() == Opcode::OP_SLICE) {
        return ShouldSkipSliceUpgradeForOffsets(op, sourceType, targetType, reason, skip);
    }
    if (op.GetOpcode() == Opcode::OP_CONTRACT) {
        return ShouldSkipContractUpgradeForOffsets(op, sourceType, targetType, reason, skip);
    }
    return SUCCESS;
}

// 等形状候选由单个 CONTRACT 写入，按 GetShape() 比较 CONTRACT 输入、中间张量及各 SLICE 输出的静态形状。
static bool IsEqualShapeContractSlice(const Operation& contract)
{
    if (contract.iOperand.size() != 1 || contract.oOperand.size() != 1) {
        return false;
    }
    const auto& input = contract.iOperand.front();
    const auto& middle = contract.oOperand.front();
    if (input == nullptr || middle == nullptr || middle->GetProducers().size() != 1 ||
        *middle->GetProducers().begin() != &contract || middle->GetConsumers().empty()) {
        return false;
    }
    const auto& shape = middle->GetShape();
    return input->GetShape() == shape &&
           std::all_of(middle->GetConsumers().begin(), middle->GetConsumers().end(), [&shape](const Operation* slice) {
               return slice != nullptr && slice->GetOpcode() == Opcode::OP_SLICE && slice->oOperand.size() == 1 &&
                      slice->oOperand.front() != nullptr && slice->oOperand.front()->GetShape() == shape;
           });
}

// 等形状路径优先让中间张量驻留 L0C，由各 SLICE 将数据搬入 UB。
// 中间张量含负维时，由 CheckContractSliceTopology 拒绝直连升级。
Status AssignMemoryType::ProcessL0C2UBEqualShape(const std::vector<Operation*>& contracts)
{
    const std::string reason = "ProcessL0C2UBEqualShape";
    for (auto* contract : contracts) {
        if (!IsEqualShapeContractSlice(*contract)) {
            continue;
        }
        RETURN_IF_NOT_SUCCESS(TryUpgradeL0C2UBContractSlice(*contract, reason));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2UBSmallToLarge(const std::vector<Operation*>& slices)
{
    for (auto* sliceOp : slices) {
        auto& op = *sliceOp;
        bool skipUpgrade = false;
        RETURN_IF_NOT_SUCCESS(ShouldSkipDirectPathUpgradeForBothSideOffsets(op, MemoryType::MEM_L0C, MemoryType::MEM_UB,
                                                                            "ProcessL0C2UBSmallToLarge", skipUpgrade));
        if (skipUpgrade) {
            continue;
        }
        // conv 多维 tile 放开 2 维限制 (requireMatrixShape=false)，对齐校验按维数分派
        RETURN_IF_NOT_SUCCESS(TryUpgradeSliceContractPath(op, MemoryType::MEM_L0C, MemoryType::MEM_UB,
                                                          "ProcessL0C2UBSmallToLarge", false, true, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL0C2UBLargeToSmall(const std::vector<Operation*>& contracts)
{
    const std::string reason = "ProcessL0C2UBLargeToSmall";
    for (auto* contract : contracts) {
        RETURN_IF_NOT_SUCCESS(TryUpgradeL0C2UBContractSlice(*contract, reason));
    }
    return SUCCESS;
}

// 从单个 CONTRACT 尝试升级 SLICE 侧的 L0C 到 UB 直连，先检查双侧偏移，再按维数检查对齐。
Status AssignMemoryType::TryUpgradeL0C2UBContractSlice(Operation& contract, const std::string& reason)
{
    bool skipUpgrade = false;
    const auto offsetStatus = ShouldSkipDirectPathUpgradeForBothSideOffsets(contract, MemoryType::MEM_L0C,
                                                                            MemoryType::MEM_UB, reason, skipUpgrade);
    if (offsetStatus != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s failed to check L0C-to-UB offsets for CONTRACT[%d].", reason.c_str(),
                          contract.GetOpMagic());
        return offsetStatus;
    }
    if (skipUpgrade) {
        return SUCCESS;
    }
    const auto upgradeStatus = TryUpgradeSingleContractSlicePath(contract, MemoryType::MEM_L0C, MemoryType::MEM_UB,
                                                                 reason, false, true, false);
    if (upgradeStatus != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "%s failed to upgrade L0C-to-UB slices for CONTRACT[%d].",
                          reason.c_str(), contract.GetOpMagic());
    }
    return upgradeStatus;
}

Status AssignMemoryType::ProcessL12UBSmallToLarge(const std::vector<Operation*>& slices)
{
    for (auto* sliceOp : slices) {
        auto& op = *sliceOp;
        RETURN_IF_NOT_SUCCESS(TryUpgradeSliceContractPath(op, MemoryType::MEM_L1, MemoryType::MEM_UB,
                                                          "ProcessL12UBSmallToLarge", false, true, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessL12UBLargeToSmall(const std::vector<Operation*>& contracts)
{
    for (auto* contractOp : contracts) {
        auto& op = *contractOp;
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_L1, MemoryType::MEM_UB,
                                                                "ProcessL12UBLargeToSmall", false, true, false));
    }
    return SUCCESS;
}

Status AssignMemoryType::ProcessUB2L1SmallToLarge(const std::vector<Operation*>& slices)
{
    for (auto* sliceOp : slices) {
        auto& op = *sliceOp;
        RETURN_IF_NOT_SUCCESS(TryUpgradeSliceContractPath(op, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                                          "ProcessUB2L1SmallToLarge", false, false, true));
    }
    return SUCCESS;
}

bool AssignMemoryType::ShouldSkipUB2L1SmallToLarge(const LogicalTensorPtr& iOperand,
                                                   const LogicalTensorPtr& oOperand) const
{
    // UB2L1 仅支持 batch 轴全 1 的非 conv 多维 tile (batch matmul);
    const bool isConv = MemoryPathUtils::HasConvConsumer(oOperand);
    if (oOperand != nullptr && oOperand->GetShape().size() > SHAPE_DIM2 &&
        (isConv || !MemoryPathUtils::AllBatchAxesAreOne(oOperand->GetShape()))) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Reject UB-to-L1 tensor[%d]: rank>2 conv tile or non-unit batch axes.",
                          oOperand->GetMagic());
        return true;
    }
    const size_t UB_LIMIT = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                UB_THRESHOLD_NORMAL);
    size_t tensorSize = IsSliceMode() ? CalcNZTensorSize(iOperand, isConv) : CalcNZTensorSizeLegacy(iOperand);
    if (tensorSize > UB_LIMIT) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Reject UB-to-L1 tensor[%d]: storage %zu bytes exceeds UB limit %zu bytes.",
                          iOperand->GetMagic(), tensorSize, UB_LIMIT);
        return true;
    }
    // 检查 consumer slice 是否有 copy_in_mode=0 属性
    for (auto& consumerOp : oOperand->GetConsumers()) {
        if (IsModeMoveIn(mode_, consumerOp->GetOpcode())) {
            int64_t copyInModeValue = 0;
            if (consumerOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) && copyInModeValue == 0) {
                // copy_in_mode=ND2ND 标记两类视图：bias/scale（ND 布局仅由 DDR 路径支持）与 GEMV A
                // （保持 ND 布局走 TCopyUB2L1ND2ND 直连，见 GenerateMoveOp::ProcessUB2L1 的 GEMV 分支）
                int64_t consumerIsGemv = 0;
                if (!consumerOp->GetAttr<int64_t>(OpAttributeKey::isGemv, consumerIsGemv) || consumerIsGemv == 0) {
                    APASS_LOG_DEBUG_F(Elements::Operation,
                                      "Reject UB-to-L1 tensor[%d]: consumer[%d] copy_in_mode is 0.",
                                      oOperand->GetMagic(), consumerOp->GetOpMagic());
                    return true;
                }
            }
            // MXMatmul场景K轴非64对齐不支持UB2L1直连（MX补齐仅由DDR路径支持），回退DDR
            if (MemoryPathUtils::IsMxPaddingMode(*consumerOp)) {
                APASS_LOG_DEBUG_F(Elements::Operation, "Reject UB-to-L1 tensor[%d]: consumer[%d] requires MX padding.",
                                  oOperand->GetMagic(), consumerOp->GetOpMagic());
                return true;
            }
        }
    }
    return !MemoryPathUtils::CheckInnerAxisC0Size(iOperand, oOperand);
}

Status AssignMemoryType::ProcessUB2L1LargeToSmall(const std::vector<Operation*>& contracts)
{
    for (auto* contractOp : contracts) {
        auto& op = *contractOp;
        RETURN_IF_NOT_SUCCESS(TryUpgradeSingleContractSlicePath(op, MemoryType::MEM_UB, MemoryType::MEM_L1,
                                                                "ProcessUB2L1LargeToSmall", false, false, true));
    }
    return SUCCESS;
}

bool AssignMemoryType::CanUseMiddleTensorForUpgrade(const LogicalTensorPtr& middle, MemoryType targetType) const
{
    if (middle == nullptr || targetType == MemoryType::MEM_UNKNOWN || IsOutCastTensor(middle)) {
        return false;
    }
    MemoryType currentType = middle->GetMemoryTypeOriginal();
    return currentType == MemoryType::MEM_UNKNOWN || currentType == MemoryType::MEM_DEVICE_DDR ||
           currentType == targetType;
}

bool AssignMemoryType::IsOutCastTensor(const LogicalTensorPtr& tensor) const
{
    return tensor != nullptr && outcastRawMagics_.has_value() && outcastRawMagics_->count(tensor->GetRawMagic()) != 0;
}

bool AssignMemoryType::HasOnlyContractProducers(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetProducers().empty()) {
        return false;
    }
    return std::all_of(tensor->GetProducers().begin(), tensor->GetProducers().end(), [](Operation* producer) {
        return producer != nullptr && producer->GetOpcode() == Opcode::OP_CONTRACT && !producer->iOperand.empty() &&
               !producer->oOperand.empty() &&
               std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute()) != nullptr;
    });
}

bool AssignMemoryType::HasOnlySliceConsumers(const LogicalTensorPtr& tensor) const
{
    if (tensor == nullptr || tensor->GetConsumers().empty()) {
        return false;
    }
    return std::all_of(tensor->GetConsumers().begin(), tensor->GetConsumers().end(), [](Operation* consumer) {
        return consumer != nullptr && consumer->GetOpcode() == Opcode::OP_SLICE && !consumer->iOperand.empty() &&
               !consumer->oOperand.empty() &&
               std::dynamic_pointer_cast<ViewOpAttribute>(consumer->GetOpAttribute()) != nullptr;
    });
}

bool AssignMemoryType::IsSliceOutputTarget(Operation& sliceOp, MemoryType targetType) const
{
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.oOperand.empty() || sliceOp.oOperand.front() == nullptr) {
        return false;
    }
    MemoryType outputOriginal = sliceOp.oOperand.front()->GetMemoryTypeOriginal();
    if (outputOriginal != MemoryType::MEM_UNKNOWN) {
        return outputOriginal == targetType;
    }
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    return viewOpAttribute != nullptr && viewOpAttribute->GetTo() == targetType;
}

Status AssignMemoryType::EnsureSliceOutputTarget(Operation& sliceOp, MemoryType targetType, const std::string& reason)
{
    if (sliceOp.oOperand.empty() || sliceOp.oOperand.front() == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because slice output is invalid.", reason.c_str(),
                          sliceOp.GetOpMagic());
        return FAILED;
    }
    auto output = sliceOp.oOperand.front();
    MemoryType outputOriginal = output->GetMemoryTypeOriginal();
    if (outputOriginal != MemoryType::MEM_UNKNOWN && outputOriginal != targetType) {
        APASS_LOG_ERROR_F(Elements::Operation,
                          "Upgrade %s[%d] failed because slice output tensor[%d] original %s conflicts with target %s.",
                          reason.c_str(), sliceOp.GetOpMagic(), output->GetMagic(),
                          BriefMemoryTypeToString(outputOriginal).c_str(), BriefMemoryTypeToString(targetType).c_str());
        return FAILED;
    }
    RETURN_IF_NOT_SUCCESS(MemoryPathUtils::SetOriginalChecked(output, targetType, reason));
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(sliceOp.GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because slice attr is null.", reason.c_str(),
                          sliceOp.GetOpMagic());
        return FAILED;
    }
    viewOpAttribute->SetToType(targetType);
    return SUCCESS;
}

Status AssignMemoryType::ApplySliceContractUpgrade(Operation& sliceOp, MemoryType sourceType, MemoryType targetType,
                                                   const std::string& reason)
{
    auto middle = sliceOp.iOperand.front();
    MemoryPathUtils::ForceSetOriginal(middle, targetType, reason);
    for (auto* consumer : middle->GetConsumers()) {
        RETURN_IF_NOT_SUCCESS(EnsureSliceOutputTarget(*consumer, targetType, reason));
        MemoryPathUtils::ForceSetRequirement(inserter, middle, *consumer, targetType, reason);
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract input is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, *producer, sourceType, reason);
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (assembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        assembleOpAttribute->SetFromType(sourceType);
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Upgrade middle tensor[%d] to %s for %s contract-side special path.",
                      middle->GetMagic(), BriefMemoryTypeToString(targetType).c_str(), reason.c_str());
    return SUCCESS;
}

Status AssignMemoryType::ApplySingleContractSliceUpgrade(Operation& contractOp, MemoryType sourceType,
                                                         MemoryType targetType, const std::string& reason)
{
    auto middle = contractOp.oOperand.front();
    MemoryPathUtils::ForceSetOriginal(middle, sourceType, reason);
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(contractOp.GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.", reason.c_str(),
                          contractOp.GetOpMagic());
        return FAILED;
    }
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (input == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract input is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, *producer, sourceType, reason);
        auto producerAssembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (producerAssembleOpAttribute == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Upgrade %s[%d] failed because contract attr is null.",
                              reason.c_str(), producer->GetOpMagic());
            return FAILED;
        }
        producerAssembleOpAttribute->SetFromType(sourceType);
    }
    for (auto* consumer : middle->GetConsumers()) {
        RETURN_IF_NOT_SUCCESS(EnsureSliceOutputTarget(*consumer, targetType, reason));
        MemoryPathUtils::ForceSetRequirement(inserter, middle, *consumer, sourceType, reason);
    }
    APASS_LOG_DEBUG_F(Elements::Tensor, "Upgrade middle tensor[%d] to %s for %s slice-side special path.",
                      middle->GetMagic(), BriefMemoryTypeToString(sourceType).c_str(), reason.c_str());
    return SUCCESS;
}

bool AssignMemoryType::CanUseL0C2L1UpgradePath(Operation& operation)
{
    if (operation.iOperand.empty() || operation.oOperand.empty() || operation.iOperand.front() == nullptr ||
        operation.oOperand.front() == nullptr) {
        return false;
    }
    return inserter.FitL0C2L1(operation);
}

// 检查中间张量和切片输出的目标类型、形状与容量，动态中间张量保留原路径。
bool AssignMemoryType::CheckSliceContractMiddle(const SliceContractPath& path)
{
    constexpr size_t kMatrixShapeDimCount = 2;
    const auto& middle = path.middle;
    const auto& target = path.target;
    const auto& sliceOp = path.sliceOp;
    const auto sourceType = path.sourceType;
    const auto targetType = path.targetType;
    const auto& reason = path.reason;
    // 与 TryUpgradeSingleContractSlicePath 一致：middle 的 shape 存在 -1（动态维度）时不参与升级，
    // 动态 shape 的 tensor 无法作为本地 scratch 静态分配。
    const auto& middleShape = middle->GetShape();
    if (std::any_of(middleShape.begin(), middleShape.end(), [](int64_t dim) { return dim < 0; })) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Skip %s: op[%d] middle tensor[%d], dynamic middle shape.",
                          reason.c_str(), sliceOp.GetOpMagic(), middle->GetMagic());
        return false;
    }
    for (auto* consumer : middle->GetConsumers()) {
        if (consumer != nullptr && !IsSliceOutputTarget(*consumer, targetType)) {
            return false;
        }
    }
    if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
        !AreAllSliceConsumerShapesPreserved(middle)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip %s: op[%d] middle tensor[%d], slice changes shape on L0C-to-L1 path.", reason.c_str(),
                          sliceOp.GetOpMagic(), middle->GetMagic());
        return false;
    }
    if (path.requireMatrixShape &&
        (middle->GetShape().size() != kMatrixShapeDimCount || target->GetShape().size() != kMatrixShapeDimCount)) {
        return false;
    }
    if (path.checkUbTileShape && !MemoryPathUtils::FitsAssembleOutputMemoryLimit(middle, targetType)) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Skip %s: op[%d] middle tensor[%d], %zu bytes exceeds %s limit %zu bytes.",
            reason.c_str(), sliceOp.GetOpMagic(), middle->GetMagic(), static_cast<size_t>(middle->GetDataSize()),
            BriefMemoryTypeToString(targetType).c_str(),
            static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(targetType) *
                                (targetType == MemoryType::MEM_UB ? UB_THRESHOLD_ASSEMBLE : L1_THRESHOLD)));
        return false;
    }
    return true;
}

// 检查单个 CONTRACT 输入的驻留类型、数据类型及直连形状约束。
bool AssignMemoryType::CheckSliceContractProducer(const SliceContractPath& path, Operation& producer,
                                                  const LogicalTensorPtr& input, bool isLiteNpu)
{
    constexpr size_t kMatrixShapeDimCount = 2;
    const auto sourceType = path.sourceType;
    const auto targetType = path.targetType;
    if (input == nullptr || input->GetMemoryTypeOriginal() != sourceType) {
        return false;
    }
    if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_UB &&
        !(MemoryPathUtils::IsL0C2UbSupportedShape(input) ||
          (isLiteNpu && MemoryPathUtils::AllBatchAxesAreOne(input->GetShape())))) {
        return false;
    }
    if (!IsContractSliceSourceDtypeSupported(input, sourceType, targetType, isLiteNpu)) {
        return false;
    }
    if (path.requireMatrixShape && input->GetShape().size() != kMatrixShapeDimCount) {
        return false;
    }
    if (path.checkUbTileShape && !CheckUBTileShapeByRank(input)) {
        return false;
    }
    if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 && !CanUseL0C2L1UpgradePath(producer)) {
        return false;
    }
    return true;
}

// 检查 L0C 到 L1 的切片约束，以及 UB 输出上会阻止升级的并行消费需求。
bool AssignMemoryType::CheckSliceContractConsumers(const SliceContractPath& path)
{
    const auto& middle = path.middle;
    const auto targetType = path.targetType;
    const auto sourceType = path.sourceType;
    const auto& reason = path.reason;
    const auto& sliceOp = path.sliceOp;
    for (auto* consumer : middle->GetConsumers()) {
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*consumer)) {
            return false;
        }
        if (targetType != MemoryType::MEM_UB) {
            continue;
        }
        auto output = consumer->oOperand.front();
        if (HasDifferentConsumerRequirement(output, targetType)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Skip %s: op[%d] middle tensor[%d], parallel output requirement conflict.",
                              reason.c_str(), sliceOp.GetOpMagic(), middle->GetMagic());
            return false;
        }
    }
    return true;
}

// 检查输入的并行需求及各级分块的倍数关系，必要时追加 UB 到 L1 的限制。
bool AssignMemoryType::CheckSliceContractInputCompatibility(const SliceContractPath& path,
                                                            const LogicalTensorPtr& input)
{
    const auto& middle = path.middle;
    const auto& target = path.target;
    const auto targetType = path.targetType;
    const auto& reason = path.reason;
    const auto& sliceOp = path.sliceOp;
    if (HasParallelDifferentConsumerRequirement(input, targetType) ||
        !MemoryPathUtils::IsDimMultiple(middle->GetShape(), input->GetShape()) ||
        !CheckConsumerMoveInShapeMultiple(middle, input) ||
        !MemoryPathUtils::IsDimMultiple(target->GetShape(), input->GetShape())) {
        APASS_LOG_DEBUG_F(
            Elements::Operation,
            "Skip %s: op[%d] middle tensor[%d], parallel input requirement conflict or incompatible tile shapes.",
            reason.c_str(), sliceOp.GetOpMagic(), middle->GetMagic());
        return false;
    }
    if (path.checkUb2L1Constraints && ShouldSkipUB2L1SmallToLarge(input, middle)) {
        return false;
    }
    return true;
}

// 按原图顺序校验中间张量及其生产、消费分支，全部满足时才将路径升级为直连。
Status AssignMemoryType::TryUpgradeSliceContractPath(Operation& sliceOp, MemoryType sourceType, MemoryType targetType,
                                                     const std::string& reason, bool requireMatrixShape,
                                                     bool checkUbTileShape, bool checkUb2L1Constraints)
{
    if (sliceOp.GetOpcode() != Opcode::OP_SLICE || sliceOp.iOperand.empty() || sliceOp.oOperand.empty()) {
        return SUCCESS;
    }
    auto middle = sliceOp.iOperand.front();
    auto target = sliceOp.oOperand.front();
    if (middle == nullptr || target == nullptr || !CanUseMiddleTensorForUpgrade(middle, targetType) ||
        !HasOnlyContractProducers(middle) || !HasOnlySliceConsumers(middle) ||
        !IsSliceOutputTarget(sliceOp, targetType)) {
        return SUCCESS;
    }
    const SliceContractPath path{sliceOp,
                                 middle,
                                 target,
                                 sourceType,
                                 targetType,
                                 reason,
                                 requireMatrixShape,
                                 checkUbTileShape,
                                 checkUb2L1Constraints};
    if (!CheckSliceContractMiddle(path)) {
        return SUCCESS;
    }
    const bool isLiteNpu = IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch());
    for (auto* producer : middle->GetProducers()) {
        auto input = producer->iOperand.front();
        if (!CheckSliceContractProducer(path, *producer, input, isLiteNpu) || !CheckSliceContractConsumers(path) ||
            !CheckSliceContractInputCompatibility(path, input)) {
            return SUCCESS;
        }
    }
    return ApplySliceContractUpgrade(sliceOp, sourceType, targetType, reason);
}

// 匹配单 CONTRACT 到 SLICE 的直连路径，按兼容切片范围执行完整或部分升级。
Status AssignMemoryType::TryUpgradeSingleContractSlicePath(Operation& contractOp, MemoryType sourceType,
                                                           MemoryType targetType, const std::string& reason,
                                                           bool requireMatrixShape, bool checkUbTileShape,
                                                           bool checkUb2L1Constraints)
{
    if (contractOp.GetOpcode() != Opcode::OP_CONTRACT || contractOp.iOperand.empty() || contractOp.oOperand.empty()) {
        return SUCCESS;
    }
    ContractSlicePath path{contractOp,
                           contractOp.iOperand.front(),
                           contractOp.oOperand.front(),
                           sourceType,
                           targetType,
                           reason,
                           requireMatrixShape,
                           checkUbTileShape,
                           checkUb2L1Constraints};
    if (!CheckContractSliceTopology(path)) {
        return SUCCESS;
    }
    const bool isLiteNpu = IsLiteNPU(Platform::Instance().GetSoc().GetNPUArch());
    if (!CheckContractSliceStorage(path, isLiteNpu) || !CheckContractSliceProducers(path, isLiteNpu)) {
        return SUCCESS;
    }
    std::vector<Operation*> compatibleSlices;
    if (!CollectCompatibleContractSlices(path, compatibleSlices)) {
        return SUCCESS;
    }
    if (compatibleSlices.size() == path.middle->GetConsumers().size()) {
        return ApplySingleContractSliceUpgrade(contractOp, sourceType, targetType, reason);
    }
    return TryPartialContractSliceUpgrade(path, compatibleSlices);
}

bool AssignMemoryType::CheckContractSliceTopology(const ContractSlicePath& path) const
{
    const auto& input = path.input;
    const auto& middle = path.middle;
    const auto sourceType = path.sourceType;
    const auto targetType = path.targetType;
    if (input == nullptr || middle == nullptr || input->GetMemoryTypeOriginal() != sourceType ||
        !CanUseMiddleTensorForUpgrade(middle, sourceType) || !HasOnlyContractProducers(middle) ||
        !HasOnlySliceConsumers(middle)) {
        return false;
    }
    // 中间张量需要静态分配，包含动态维度时跳过升级。
    const auto& middleShape = middle->GetShape();
    if (std::any_of(middleShape.begin(), middleShape.end(), [](int64_t dim) { return dim < 0; })) {
        APASS_LOG_DEBUG_F(Elements::Operation, "Skip %s: op[%d] middle tensor[%d], dynamic middle shape.",
                          path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
        return false;
    }
    // 跨内存大搬小要求单个 CONTRACT，避免多路写入偏移与 SLICE 读取偏移叠加。
    // UB2UB 等同内存通路允许多个 CONTRACT 汇入，由调用方逐一检查生产者。
    if (sourceType != targetType &&
        (middle->GetProducers().size() != 1 || *middle->GetProducers().begin() != &path.contract)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip %s: op[%d] middle tensor[%d], multiple contract producers on a cross-memory path.",
                          path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
        return false;
    }
    return true;
}

bool AssignMemoryType::CheckContractSliceStorage(ContractSlicePath& path, bool isLiteNpu) const
{
    const auto& middle = path.middle;
    const auto& middleShape = middle->GetShape();
    // 大搬小升级后，中间张量驻留在源内存。UB 容量计入 ND 原数据和 NZ 副本，L0C 只计 NZ。
    // 同内存通路由调用方检查容量，且不会在中间张量上产生 NZ 副本。
    if (path.sourceType == path.targetType || middleShape.size() < SHAPE_DIM2) {
        return true;
    }
    if (path.sourceType == MemoryType::MEM_UB) {
        // 卷积和普通 Matmul 的 UB2L1 通路要求中间张量至少为二维，低于二维的早退有意保留 isConvConsumer 为 false
        path.isConvConsumer = MemoryPathUtils::HasConvConsumer(middle);
        if (middleShape.size() > SHAPE_DIM2 &&
            (path.isConvConsumer || !MemoryPathUtils::AllBatchAxesAreOne(middleShape))) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Skip %s: op[%d] middle tensor[%d], rank>2 conv tile or non-unit batch axes.",
                              path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
            return false;
        }
        const size_t ubLimit = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                                   UB_THRESHOLD_NORMAL);
        if (CalcNZTensorSize(middle, path.isConvConsumer) > ubLimit) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Skip %s: op[%d] middle tensor[%d], storage %zu bytes exceeds UB limit %zu bytes.",
                              path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic(),
                              CalcNZTensorSize(middle, path.isConvConsumer), ubLimit);
            return false;
        }
    } else if (path.sourceType == MemoryType::MEM_L0C) {
        path.isConvProducer = MemoryPathUtils::HasConvProducer(path.input);
        // 高维直连支持卷积，以及 LiteNPU 上所有批次维度均为 1 的批量矩阵乘。
        if (middleShape.size() > SHAPE_DIM2 && !path.isConvProducer &&
            !(isLiteNpu && MemoryPathUtils::AllBatchAxesAreOne(middleShape))) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Skip %s: op[%d] middle tensor[%d], rank>2 non-unit batch axes.",
                              path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
            return false;
        }
        const size_t l0cLimit = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C) *
                                                    L0C_THRESHOLD);
        if (CalcNzStorageSize(middle, path.isConvProducer) > l0cLimit) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Skip %s: op[%d] middle tensor[%d], storage %zu bytes exceeds L0C limit %zu bytes.",
                              path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic(),
                              CalcNzStorageSize(middle, path.isConvProducer), l0cLimit);
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::CheckContractSliceProducers(const ContractSlicePath& path, bool isLiteNpu)
{
    const auto& input = path.input;
    const auto& middle = path.middle;
    const auto sourceType = path.sourceType;
    const auto targetType = path.targetType;
    // 先检查输入和中间张量的并行需求，再逐个检查生产者的通路约束。
    if (path.requireMatrixShape && input->GetShape().size() != SHAPE_DIM2) {
        return false;
    }
    if (HasParallelDifferentConsumerRequirement(input, targetType)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip %s: op[%d] middle tensor[%d], parallel input requirement conflict.",
                          path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
        return false;
    }
    if (HasParallelDifferentConsumerRequirement(middle, targetType)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip %s: op[%d] middle tensor[%d], parallel middle requirement conflict.",
                          path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
        return false;
    }
    for (auto* producer : middle->GetProducers()) {
        auto producerInput = producer->iOperand.front();
        if (producerInput == nullptr || producerInput->GetMemoryTypeOriginal() != sourceType) {
            return false;
        }
        if (!IsContractSliceSourceDtypeSupported(producerInput, sourceType, targetType, isLiteNpu)) {
            return false;
        }
        if (path.requireMatrixShape && producerInput->GetShape().size() != SHAPE_DIM2) {
            return false;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*producer)) {
            return false;
        }
        if (HasParallelDifferentConsumerRequirement(producerInput, targetType)) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "Skip %s: op[%d] middle tensor[%d], parallel producer requirement conflict.",
                              path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::IsContractSliceSourceDtypeSupported(const LogicalTensorPtr& input, MemoryType sourceType,
                                                           MemoryType targetType, bool isLiteNpu) const
{
    if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_UB) {
        return isLiteNpu ? inserter.IsL0C2UbLiteNPUSupportedDtype(input) : inserter.IsL0C2UbSupportedDtype(input);
    }
    if (isLiteNpu && sourceType == MemoryType::MEM_L1 && targetType == MemoryType::MEM_UB) {
        return inserter.IsL12UbLiteNPUSupportedDtype(input);
    }
    if (sourceType == MemoryType::MEM_UB && targetType == MemoryType::MEM_L1) {
        return isLiteNpu ? inserter.IsUb2L1LiteNPUSupportedDtype(input) : inserter.IsUb2L1SupportedDtype(input);
    }
    return true;
}

bool AssignMemoryType::CollectCompatibleContractSlices(const ContractSlicePath& path,
                                                       std::vector<Operation*>& compatibleSlices)
{
    const auto& input = path.input;
    const auto& middle = path.middle;
    const auto sourceType = path.sourceType;
    const auto targetType = path.targetType;
    const bool allowPartialUpgrade = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 &&
                                     MemoryPathUtils::IsSpecialDirectMemoryPath(sourceType, targetType) &&
                                     !MemoryPathUtils::HasConvProducer(input);
    // A5 的部分升级只允许倍数关系不符的分支回退，其他检查仍适用于所有 Slice。
    for (auto* consumer : middle->GetConsumers()) {
        if (!IsSliceOutputTarget(*consumer, targetType)) {
            return false;
        }
        auto output = consumer->oOperand.front();
        if (path.requireMatrixShape && output->GetShape().size() != SHAPE_DIM2) {
            return false;
        }
        if (sourceType == MemoryType::MEM_L0C && targetType == MemoryType::MEM_L1 &&
            !CanUseL0C2L1UpgradePath(*consumer)) {
            return false;
        }
        bool shapeCompatible = MemoryPathUtils::IsDimMultiple(input->GetShape(), output->GetShape());
        if (sourceType != MemoryType::MEM_L0C || targetType != MemoryType::MEM_L1) {
            shapeCompatible = shapeCompatible || MemoryPathUtils::IsDimMultiple(output->GetShape(), input->GetShape());
        }
        if (shapeCompatible) {
            compatibleSlices.push_back(consumer);
        } else {
            if (!allowPartialUpgrade) {
                APASS_LOG_DEBUG_F(Elements::Operation,
                                  "Skip %s: op[%d] middle tensor[%d], incompatible input and output tile shapes.",
                                  path.reason.c_str(), path.contract.GetOpMagic(), middle->GetMagic());
                return false;
            }
            APASS_LOG_DEBUG_F(Elements::Operation,
                              "%s: Contract[%d] keeps Slice[%d] on the original path due to incompatible tile shapes.",
                              path.reason.c_str(), path.contract.GetOpMagic(), consumer->GetOpMagic());
        }
        if (path.checkUbTileShape && !CheckUBTileShapeByRank(output)) {
            return false;
        }
        if (path.checkUb2L1Constraints && !CheckContractSliceUB2L1(path, *consumer)) {
            return false;
        }
    }
    return true;
}

bool AssignMemoryType::CheckContractSliceUB2L1(const ContractSlicePath& path, const Operation& slice) const
{
    const size_t ubLimit = static_cast<size_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) *
                                               UB_THRESHOLD_NORMAL);
    int64_t copyInModeValue = 0;
    int64_t consumerIsGemv = 0;
    // GEMV 的 A 输入允许走 ND2ND 直连，偏置和缩放因子的 ND2ND 搬运仍经 DDR。
    const bool ndCopyInReject = slice.GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue) &&
                                copyInModeValue == 0 &&
                                !(slice.GetAttr<int64_t>(OpAttributeKey::isGemv, consumerIsGemv) &&
                                  consumerIsGemv != 0);
    // MXMatmul 的 K 轴未按 64 对齐时需要经 DDR 补齐。
    if (CalcNZTensorSize(path.input, path.isConvConsumer) > ubLimit || ndCopyInReject ||
        MemoryPathUtils::IsMxPaddingMode(slice)) {
        APASS_LOG_DEBUG_F(Elements::Operation,
                          "Skip %s: op[%d] middle tensor[%d], UB-to-L1 size, copy mode or MX padding constraints.",
                          path.reason.c_str(), path.contract.GetOpMagic(), path.middle->GetMagic());
        return false;
    }
    return true;
}

Status AssignMemoryType::TryPartialContractSliceUpgrade(const ContractSlicePath& path,
                                                        const std::vector<Operation*>& compatibleSlices)
{
    const auto& input = path.input;
    const auto& middle = path.middle;
    const auto sourceType = path.sourceType;
    const auto targetType = path.targetType;
    auto middleType = middle->GetMemoryTypeOriginal();
    if (compatibleSlices.empty() ||
        (middleType != MemoryType::MEM_UNKNOWN && middleType != MemoryType::MEM_DEVICE_DDR) ||
        !IsWholeTensorContract(path.contract) || inserter.GetRequirementOrUnknown(input, path.contract) != sourceType ||
        !std::all_of(compatibleSlices.begin(), compatibleSlices.end(), [&middle](const Operation* slice) {
            return slice->iOperand.size() == 1 && slice->iOperand.front() == middle && slice->oOperand.size() == 1;
        })) {
        return SUCCESS;
    }

    // 剩余 Slice 继续经原 CONTRACT 读取 DDR，改接的 Slice 直接从源内存搬运。
    const auto sliceCount = middle->GetConsumers().size();
    auto middleRequirements = inserter.tensorTobeMap.find(middle);
    for (auto* slice : compatibleSlices) {
        RETURN_IF_NOT_SUCCESS(EnsureSliceOutputTarget(*slice, targetType, path.reason));
        slice->ReplaceIOperand(0, input);
        if (middleRequirements != inserter.tensorTobeMap.end()) {
            middleRequirements->second.erase(slice->GetOpMagic());
        }
        MemoryPathUtils::ForceSetRequirement(inserter, input, *slice, sourceType, path.reason);
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "%s: Contract[%d] rewired %zu/%zu slices for direct transfer.",
                      path.reason.c_str(), path.contract.GetOpMagic(), compatibleSlices.size(), sliceCount);
    return SUCCESS;
}

size_t AssignMemoryType::CalcNzAlignedSize(const LogicalTensorPtr& tensor, bool isConv) const
{
    constexpr int64_t kC0AlignBytes = 32;
    const auto& shape = tensor->GetShape();
    if (shape.size() < 2) {
        return 0;
    }
    DataType dtype = tensor->Datatype();
    int64_t bytes = BytesOf(dtype);
    auto [outer, inner] = GetNzOuterInnerAxes(shape, isConv);
    // 外轴对齐：INT8/FP8 对齐到 32，其他对齐到 16
    size_t outerAlign = (dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_FP8) ? 32 : 16;
    // 内轴对齐：C0 size = 32 / 元素字节数
    size_t c0 = 0;
    if (bytes > 0) {
        c0 = static_cast<size_t>(kC0AlignBytes / bytes);
    }
    if (c0 == 0) {
        APASS_LOG_DEBUG_F(Elements::Operation, "CalcNzAlignedSize: invalid C0 size, c0=%zu", c0);
        return 0;
    }
    size_t alignedOuter = (outer + outerAlign - 1) / outerAlign * outerAlign + 1;
    size_t alignedInner = (inner + c0 - 1) / c0 * c0;
    return alignedOuter * alignedInner * static_cast<size_t>(bytes);
}

size_t AssignMemoryType::CalcNZTensorSize(const LogicalTensorPtr& tensor, bool isConv) const
{
    constexpr int64_t kAlignBytes = 4;
    const auto& shape = tensor->GetShape();
    if (shape.size() < 2) {
        return 0;
    }
    int64_t bytes = BytesOf(tensor->Datatype());
    // ND 为全量元素数 (全部维度乘积): conv 多维 tile 不能只取末两维, 否则丢失 N/C/D 前导维
    size_t numel = ShapeNumel(shape);
    size_t ndSize = numel * static_cast<size_t>(bytes);
    size_t nzSize = CalcNzAlignedSize(tensor, isConv);
    if (nzSize == 0) {
        return numel * static_cast<size_t>(bytes > 0 ? bytes : kAlignBytes);
    }
    return ndSize + nzSize;
}

size_t AssignMemoryType::CalcNzStorageSize(const LogicalTensorPtr& tensor, bool isConv) const
{
    constexpr int64_t kAlignBytes = 4;
    const auto& shape = tensor->GetShape();
    if (shape.size() < 2) {
        return 0;
    }
    int64_t bytes = BytesOf(tensor->Datatype());
    size_t nzSize = CalcNzAlignedSize(tensor, isConv);
    if (nzSize == 0) {
        // C0 无法推导时退化为 ND 全量大小 (全部维度乘积)
        return ShapeNumel(shape) * static_cast<size_t>(bytes > 0 ? bytes : kAlignBytes);
    }
    return nzSize;
}
} // namespace npu::tile_fwk
