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
 * \file memory_path_utils.h
 * \brief
 */

#ifndef TILE_FWK_MEMORY_PATH_UTILS_H
#define TILE_FWK_MEMORY_PATH_UTILS_H

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/checker_utils.h"
#include "passes/tile_graph_pass/data_path/assign_memory_type_mode.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"

// 为共用的内存路径实现提供统一的日志模块名。
#ifndef MODULE_NAME
#define MODULE_NAME "AssignMemoryType"
#endif
#ifndef RETURN_IF_NOT_SUCCESS
#define RETURN_IF_NOT_SUCCESS(expr)                \
    do {                                           \
        Status assignMemoryReturnStatus = (expr);  \
        if (assignMemoryReturnStatus != SUCCESS) { \
            return assignMemoryReturnStatus;       \
        }                                          \
    } while (0)
#endif

namespace npu::tile_fwk {

class ConvertInserter;

// kirin9050 matmul 宏指令 opcode 判定（与通用 matmul opcode 判定解耦，供 pass 独立处理）
static inline bool IsMatmulMacroInstrOpcode(Opcode opcode)
{
    static const std::set<Opcode> kMatmulMacroInstrOps = {Opcode::OP_A_MUL_B_EPILOGUE,  Opcode::OP_A_MULACC_B_EPILOGUE,
                                                          Opcode::OP_A_MUL_BT_EPILOGUE, Opcode::OP_A_MULACC_BT_EPILOGUE,
                                                          Opcode::OP_AT_MUL_B_EPILOGUE, Opcode::OP_AT_MUL_BT_EPILOGUE};
    return kMatmulMacroInstrOps.count(opcode) > 0;
}

class MemoryPathUtils {
public:
    using OutputRequirementResolver = std::function<MemoryType(const LogicalTensorPtr&)>;

    static bool IsSpecialDirectMemoryPath(MemoryType from, MemoryType to);

    static bool IsDifferentKnownRequirement(MemoryType requirement, MemoryType targetType);

    static bool ShouldUseDdrForSpecialPath(bool hasParallelDifferentRequirement, MemoryType from, MemoryType to);

    // UB2L1约束：当前op的copy_in_l1_padding_mode为MX_PADDING_MODE时，取拷入源tensor（view输入）
    // 的dynValidShape（validK，K维由copy_in_l1_k_index指示位于第0/1维）：validK非concrete直接
    // 回退经DDR搬运（MX补齐语义仅由DDR路径TLoad支持）；validK为concrete且按64（MX scale块
    // 大小）对齐时无需K向补齐，允许UB->L1直连，否则同样回退DDR。
    static bool IsMxPaddingMode(const Operation& operation);

    // 判断tensor的消费者中是否存在需回退DDR的MX_PADDING_MODE拷入op（validK非concrete或
    // 未按64对齐），用于UB2L1路径回退DDR。
    static bool HasMxPaddingModeConsumer(const LogicalTensorPtr& tensor);

    // 在判断直接搬运路径冲突前，解析 view/assemble 语义 op 背后的有效消费者需求。
    // OP_VIEW/OP_SLICE: 通过 ViewOpAttribute.GetTo() 获取目的地类型。
    // OP_ASSEMBLE/OP_CONTRACT: 若处于特殊直连路径且 output 有下游消费者，推断 targetType；
    //                          无下游消费者（如 outcast）则保留 output MemoryTypeOriginal。
    // resolver 的语义与阶段相关：AssignMemoryType 阶段可递归穿透 view 消费者，
    // ConvertInserter 阶段应基于 tensorTobeMap 中已经规划好的直接需求判断。
    static MemoryType ResolveEffectiveConsumerRequirement(Operation* consumerOp, MemoryType directRequirement,
                                                          MemoryType targetType,
                                                          const OutputRequirementResolver& resolveOutputRequirement);

    static Status GetFirstInputOutputIfOpcode(Operation& operation, Opcode expectedOpcode, const std::string& action,
                                              LogicalTensorPtr& input, LogicalTensorPtr& output, bool& shouldHandle);

    // 特殊进阶数据通路，不满足特定条件时回退到通过DDR搬运：L0C2L1, L0C2UB, UB2L1
    static bool IsAdvancedMemoryPath(MemoryType from, MemoryType to);

    static bool IsAssembleProducer(Operation* operation);

    static void CollectProducerAIVFlags(Operation* op, std::vector<bool>& isProducerVector);

    static void CollectConsumerAICFlags(Operation* op, std::vector<bool>& isConsumerCube);

    static bool FitsAssembleOutputMemoryLimit(const LogicalTensorPtr& output, MemoryType memoryType);

    static bool FitsTensorInUb(const LogicalTensorPtr& tensor);

    static bool ExceedsMemoryLimit(const LogicalTensorPtr& tensor, size_t threshold);

    static bool IsOversizedLocalBuffer(const LogicalTensorPtr& tensor, MemoryType memoryType, bool useStrictUbLimit,
                                       bool allowL1Fallback);

    static bool IsDynamicReshape(Operation& operation, const LogicalTensorPtr& output);

    static bool HasConvProducer(const LogicalTensorPtr& tensor);

    static bool HasConvConsumer(const LogicalTensorPtr& tensor);

    static bool AllBatchAxesAreOne(const Shape& shape);

    static bool IsL0C2UbSupportedShape(const LogicalTensorPtr& tensor);

    static bool CheckUBTileShape(const LogicalTensorPtr& moveTensor);

    static bool CheckUBConvTileShape(const LogicalTensorPtr& moveTensor);

    // LiteNPU batch matmul 分形校验 (N轴对齐)
    static bool CheckUBBatchMatmulTileShapeLiteNPU(const LogicalTensorPtr& moveTensor);

    static bool IsDimMultiple(const Shape& shape1, const Shape& shape2);

    static bool CheckInnerAxisC0Size(const LogicalTensorPtr& input, const LogicalTensorPtr& output);

    static Status SyncTensorToBe(Function& function);

    static Status SetOriginalChecked(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                     const std::string& reason = "unknown", bool allowOverride = false);

    static void ForceSetOriginal(const LogicalTensorPtr& tensor, MemoryType memoryType,
                                 const std::string& reason = "unknown");

    static bool CanUseUbForReshape(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                   MemoryType inputRequirement, MemoryType outputOriginal);

    static Status AssignInOutCastMemoryTypes(Function& function);

    static Status SetRequirementChecked(ConvertInserter& inserter, const LogicalTensorPtr& tensor, Operation& operation,
                                        MemoryType memoryType, const std::string& reason, bool allowOverride = false);

    static void ForceSetRequirement(ConvertInserter& inserter, const LogicalTensorPtr& tensor, Operation& operation,
                                    MemoryType memoryType, const std::string& reason);

    static void FillUnknownRequirementsWith(ConvertInserter& inserter, const LogicalTensorPtr& tensor,
                                            MemoryType memoryType, const char* reason);

    static Status EnsureAllConsumerRequirementsExist(ConvertInserter& inserter, Function& function);

    static Status HandleNopMemoryType(ConvertInserter& inserter, Operation& operation);

    static Status HandleScatterInplaceMemoryType(ConvertInserter& inserter, Function& function, Operation& operation);

    static Status ApplyOtherSpecialOpcodeRules(ConvertInserter& inserter, Function& function);

    static Status AssignReduceAccInputRequirements(ConvertInserter& inserter, Operation& operation);

    static Status AssignOpcodeDefinedMemoryTypes(ConvertInserter& inserter, Operation& operation);

    static MemoryType GetAssembleInputType(const ConvertInserter& inserter, Operation& operation);

    static MemoryType GetReshapeInputRequirement(ConvertInserter& inserter, Operation& operation,
                                                 const LogicalTensorPtr& input, MemoryType inputOriginal);

    static Status InferReshapeOutputFromRequirement(ConvertInserter& inserter, const LogicalTensorPtr& output,
                                                    MemoryType& outputOriginal);

    static MemoryType InferUniqueRequirementThroughViewConsumers(const ConvertInserter& inserter,
                                                                 const LogicalTensorPtr& tensor);

    static MemoryType InferUniqueRequirementThroughViewConsumers(
        const ConvertInserter& inserter, const LogicalTensorPtr& tensor,
        std::unordered_set<const LogicalTensor*>& visitedTensors);

    static bool HasRequirementThroughViewConsumers(const ConvertInserter& inserter, const LogicalTensorPtr& tensor,
                                                   MemoryType targetRequirement,
                                                   std::unordered_set<const LogicalTensor*>& visitedTensors);

    static Status ApplyReshapeMemoryType(ConvertInserter& inserter, Operation& operation, const LogicalTensorPtr& input,
                                         const LogicalTensorPtr& output, bool isDynamic, bool canUseUb);

    static Status InferViewTypeInput(ConvertInserter& inserter, Operation& operation, const LogicalTensorPtr& input,
                                     const LogicalTensorPtr& output, MemoryType targetType);

    static MemoryType InferTargetTypeThroughForwardViews(const ConvertInserter& inserter,
                                                         const LogicalTensorPtr& tensor);

    static MemoryType InferTargetTypeThroughForwardViews(const ConvertInserter& inserter,
                                                         const LogicalTensorPtr& tensor,
                                                         std::unordered_set<LogicalTensorPtr>& visitedTensors);

    static Status ResolveTensorMemoryUnknowns(ConvertInserter& inserter, const LogicalTensorPtr& tensor);

    static MemoryType InferOriginalFromRequirements(const ConvertInserter& inserter, const LogicalTensorPtr& tensor);

private:
    // 与 assign_memory_type.h 中的同名常量保持一致（拆分副本）
    static constexpr double UB_THRESHOLD_ASSEMBLE = 0.35;
    static constexpr double UB_THRESHOLD_NORMAL = 1.0;
    static constexpr double L1_THRESHOLD = 0.5;
    static constexpr uint16_t L0C_TILE_SIZE = 16;
    static constexpr uint16_t INT8_ALIGN_SIZE = 32;
};

} // namespace npu::tile_fwk

#endif // TILE_FWK_MEMORY_PATH_UTILS_H
