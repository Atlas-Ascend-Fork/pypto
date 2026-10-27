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
 * \file view_reshape_assemble_reorder_utils.h
 * \brief utils of view/assemble and reshape operation reordering
 */

#ifndef PASS_VIEW_RESHAPE_ASSEMBLE_REORDER_UTILS_H_
#define PASS_VIEW_RESHAPE_ASSEMBLE_REORDER_UTILS_H_

#include <any>
#include <unordered_set>
#include <vector>

#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
using SymbolicShape = std::vector<SymbolicScalar>;
using SymbolicOffset = std::vector<SymbolicScalar>;
class ViewReshapeAssembleReorderUtils {
public:
    ViewReshapeAssembleReorderUtils() = default;
    ~ViewReshapeAssembleReorderUtils() = default;

    static Status ReorderViewReshapeAssemble(Function& function);

    static bool RemapOffsetBackwardThroughReshape(const LogicalTensorPtr& reshapeInput,
                                                  const LogicalTensorPtr& reshapeOutput, const Shape& outputBaseShape,
                                                  const SymbolicShape& outputBaseDynShape, const Offset& outputOffset,
                                                  const SymbolicOffset& outputDynOffset, Shape& inputBaseShape,
                                                  SymbolicShape& inputBaseDynShape, Offset& inputOffset,
                                                  SymbolicOffset& inputDynOffset);

    Status Process(Function& function);

private:
    struct AxisGroup {
        size_t srcBegin = 0;
        size_t srcEnd = 0;
        size_t dstBegin = 0;
        size_t dstEnd = 0;
    };

    using AxisPlan = std::vector<AxisGroup>;

    struct RemapResult {
        Offset staticOffset;
        SymbolicOffset dynOffset;
    };

    struct ChainMatch {
        LogicalTensorPtr input;
        LogicalTensorPtr middle;
        Operation* secondOp = nullptr;
        LogicalTensorPtr output;
    };

    struct ViewReshapeRecord {
        Operation* viewOp = nullptr;
        Operation* reshapeOp = nullptr;
        LogicalTensorPtr input;
        LogicalTensorPtr output;
        LogicalTensorPtr reshapeOutput;
        Offset viewOffset;
        SymbolicOffset viewDynOffset;
        SymbolicShape reshapeDynShape;
        MemoryType toType = MemoryType::MEM_UNKNOWN;
        bool hasCopyInMode = false;
        std::any copyInModeValue;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct FanoutViewRecord {
        Operation* viewOp = nullptr;
        LogicalTensorPtr output;
        Offset viewOffset;
        SymbolicOffset viewDynOffset;
        SymbolicShape outputDynShape;
        MemoryType toType = MemoryType::MEM_UNKNOWN;
        bool hasCopyInMode = false;
        std::any copyInModeValue;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct ViewReshapeFanoutRecord {
        Operation* viewOp = nullptr;
        Operation* reshapeOp = nullptr;
        LogicalTensorPtr input;
        LogicalTensorPtr reshapeOutput;
        SymbolicShape reshapeDynShape;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
        std::vector<FanoutViewRecord> fanoutViews;
    };

    struct ReshapeAssembleRecord {
        Operation* reshapeOp = nullptr;
        Operation* assembleOp = nullptr;
        LogicalTensorPtr input;
        LogicalTensorPtr output;
        LogicalTensorPtr assembleOutput;
        Offset assembleOffset;
        SymbolicOffset assembleDynOffset;
        SymbolicShape reshapeDynShape;
        SymbolicShape outputDynShape;
        MemoryType fromType = MemoryType::MEM_UNKNOWN;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct FaninAssembleRecord {
        Operation* assembleOp = nullptr;
        LogicalTensorPtr input;
        Offset assembleOffset;
        SymbolicOffset assembleDynOffset;
        SymbolicShape inputDynShape;
        MemoryType fromType = MemoryType::MEM_UNKNOWN;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
    };

    struct ReshapeAssembleFaninRecord {
        Operation* reshapeOp = nullptr;
        Operation* assembleOp = nullptr;
        LogicalTensorPtr output;
        LogicalTensorPtr reshapeInput;
        SymbolicShape reshapeDynShape;
        SymbolicShape outputDynShape;
        ir::Span span;
        Operation::ScopeInfo scopeInfo;
        std::vector<FaninAssembleRecord> faninAssembles;
    };

    void ClearRecords();
    bool HasRecords() const;
    static bool HasCascadedPattern(Function& function);
    static bool HasMixPattern(Function& function);
    Status ProcessOperations(Function& function);
    Status TryRecordViewReshape(Function& function, Operation& viewOp);
    Status TryRecordReshapeAssemble(Function& function, Operation& reshapeOp);
    Status TryRecordViewReshapeFanout(Function& function, Operation& viewOp, Operation& reshapeOp,
                                      const ChainMatch& match, const ViewOpAttribute& viewAttr,
                                      const Shape& reshapeOutputShape, const SymbolicShape& reshapeDynShape,
                                      const SymbolicShape& inputDynShape);
    Status TryCollectFanoutViewRecord(Operation& reshapeOp, Operation& consumer, const ChainMatch& match,
                                      const ViewOpAttribute& viewAttr, const SymbolicShape& compactDynShape,
                                      const SymbolicShape& middleDynShape, const SymbolicShape& inputDynShape,
                                      const Shape& reshapeOutputShape, const SymbolicShape& reshapeDynShape,
                                      FanoutViewRecord& fanoutRecord, bool& canReorder);
    Status TryRecordDirectReshapeAssemble(Function& function, Operation& reshapeOp, Operation& assembleOp,
                                          const ChainMatch& match, const AssembleOpAttribute& assembleAttr,
                                          const Shape& assembleOutputShape, const SymbolicShape& assembleDynShape,
                                          const SymbolicShape& middleDynShape, const SymbolicShape& outputDynShape);
    Status TryRecordReshapeAssembleFanin(Function& function, Operation& reshapeOp, Operation& assembleOp,
                                         const ChainMatch& match, const AssembleOpAttribute& assembleAttr,
                                         const AxisPlan& axisPlan);
    Status TryCollectFaninAssembleRecord(Operation& reshapeOp, Operation& assembleOp, Operation& producer,
                                         const ChainMatch& match, const AssembleOpAttribute& assembleAttr,
                                         const SymbolicShape& inputDynShape, const SymbolicShape& middleDynShape,
                                         const SymbolicShape& outputDynShape, const Shape& reshapeInputShape,
                                         const SymbolicShape& reshapeDynShape, FaninAssembleRecord& faninRecord,
                                         bool& canReorder);
    void AppendViewReshapeRecords(Function& function);
    void AppendViewReshapeFanoutRecords(Function& function);
    void AppendReshapeAssembleRecords(Function& function);
    void AppendReshapeAssembleFaninRecords(Function& function);
    void CleanUp(Function& function);
    void MarkViewReshapeFanoutVisited(Operation& viewOp, Operation& reshapeOp, const ViewReshapeFanoutRecord& record);
    void MarkReshapeAssembleFaninVisited(Operation& reshapeOp, Operation& assembleOp,
                                         const ReshapeAssembleFaninRecord& record);
    Operation& CreateMetadataReshape(Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                     const SymbolicShape& dynShape, const ir::Span& span,
                                     const Operation::ScopeInfo& scopeInfo, Operation& srcOp);
    static bool InferInputDynRawShapeFromOutput(const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                                                SymbolicShape& inferredInputDynRawShape);
    static bool BuildAxisPlanAllowFirstUnknown(const Shape& srcShape, const Shape& dstShape, AxisPlan& axisPlan);
    Operation& CreateView(Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                          const Offset& offset, const SymbolicOffset& dynOffset, const SymbolicShape& outputDynShape,
                          MemoryType toType, bool hasCopyInMode, const std::any& copyInModeValue, const ir::Span& span,
                          const Operation::ScopeInfo& scopeInfo);
    void CreateAssemble(Function& function, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                        const Offset& offset, const SymbolicOffset& dynOffset, const SymbolicShape& inputDynShape,
                        MemoryType fromType, const ir::Span& span, const Operation::ScopeInfo& scopeInfo,
                        Operation& srcOp);

    static bool GetChainMatch(Operation& firstOp, Opcode secondOpcode, ChainMatch& match);
    static Operation* GetPrecedingViewOp(Operation& reshapeOp);
    static Operation* GetFollowingAssembleOp(Operation& reshapeOp);
    static bool ValidateChainShapes(const ChainMatch& match);
    static bool BuildAxisPlan(const Shape& srcShape, const Shape& dstShape, AxisPlan& axisPlan);
    static bool ApplyForwardShape(const Shape& baseShape, const SymbolicShape& baseDynShape, const AxisPlan& axisPlan,
                                  Shape& newShape, SymbolicShape& newDynShape);
    static bool ApplyBackwardShape(const Shape& baseShape, const SymbolicShape& baseDynShape, const AxisPlan& axisPlan,
                                   Shape& newShape, SymbolicShape& newDynShape);
    static bool RemapOffset(const Offset& oldOffset, const SymbolicOffset& oldDynOffset, const Shape& oldShape,
                            const SymbolicShape& oldDynShape, const Shape& newShape, const SymbolicShape& newDynShape,
                            RemapResult& result);
    static bool RemapFanoutViewOffset(const Offset& baseViewOffset, const SymbolicOffset& baseViewDynOffset,
                                      const Offset& fanoutOffset, const SymbolicOffset& fanoutDynOffset,
                                      const Shape& compactShape, const SymbolicShape& compactDynShape,
                                      const Shape& middleShape, const SymbolicShape& middleDynShape,
                                      const Shape& inputShape, const SymbolicShape& inputDynShape,
                                      const Shape& newShape, const SymbolicShape& newDynShape, RemapResult& result);
    static bool RemapFaninAssembleOffset(const Offset& inputAssembleOffset,
                                         const SymbolicOffset& inputAssembleDynOffset,
                                         const Offset& outputAssembleOffset,
                                         const SymbolicOffset& outputAssembleDynOffset, const Shape& compactShape,
                                         const SymbolicShape& compactDynShape, const Shape& middleShape,
                                         const SymbolicShape& middleDynShape, const Shape& outputShape,
                                         const SymbolicShape& outputDynShape, const Shape& newShape,
                                         const SymbolicShape& newDynShape, RemapResult& result);
    static bool IsContiguousRegion(const Offset& offset, const Shape& regionShape, const Shape& baseShape);
    static bool IsLinearizedContiguousRegion(const Offset& offset, const Shape& regionShape, const Shape& baseShape);
    static bool AreCollapsedGroupsContiguous(const Offset& offset, const Shape& regionShape, const Shape& baseShape,
                                             const AxisPlan& axisPlan, bool useSrcGroup);
    static bool GetSymbolicShape(const LogicalTensorPtr& tensor, SymbolicShape& dynShape);
    static bool GetChainSymbolicShapes(const ChainMatch& match, SymbolicShape& inputDynShape,
                                       SymbolicShape& middleDynShape, SymbolicShape& outputDynShape);
    static SymbolicShape GetSymbolicShapeOrStatic(const LogicalTensorPtr& tensor);
    static SymbolicOffset NormalizeDynOffset(const Offset& offset, const SymbolicOffset& dynOffset);
    static bool BuildAssembledValidShape(const Offset& offset, const SymbolicOffset& dynOffset,
                                         const SymbolicShape& inputDynShape, size_t outputRank,
                                         SymbolicShape& outputDynShape);
    static bool MergeValidShape(const SymbolicShape& candidate, SymbolicShape& merged);
    static ir::Span GetFirstSpan(Operation& first, Operation& second);
    static Operation::ScopeInfo GetChainScopeInfo(Operation& first, Operation& second);
    static bool IsScopeCompatible(Operation& first, Operation& second);

    std::unordered_set<int> visitedOp_;
    std::vector<ViewReshapeRecord> viewReshapeRecords_;
    std::vector<ViewReshapeFanoutRecord> viewReshapeFanoutRecords_;
    std::vector<ReshapeAssembleRecord> reshapeAssembleRecords_;
    std::vector<ReshapeAssembleFaninRecord> reshapeAssembleFaninRecords_;
    IRBuilder irBuilder_;
};
} // namespace npu::tile_fwk
#endif // PASS_VIEW_RESHAPE_ASSEMBLE_REORDER_UTILS_H_
