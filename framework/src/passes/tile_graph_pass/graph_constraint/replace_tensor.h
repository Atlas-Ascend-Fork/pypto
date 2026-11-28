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
 * \file replace_tensor.h
 * \brief
 */

#ifndef REPLACE_TENSOR_H
#define REPLACE_TENSOR_H
#include <vector>
#include <climits>
#include <queue>

#include "tilefwk/data_type.h"
#include "interface/operation/opcode.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/configs/config_manager.h"

namespace npu {
namespace tile_fwk {

/*
key: Opcode类型
vaule: vector of pair, 每个pair记录了第几个输入和第几个输出存在inplace关系
*/
const std::unordered_map<Opcode, std::vector<std::pair<size_t, size_t>>> inplaceOpMap = {
    {Opcode::OP_A_MULACC_B, {std::pair<size_t, size_t>{2, 0}}},
    // kirin9050 宏指令：partial sum 与输出均为 L1，且 pto-isa TMATMUL_ACC 忽略 cIn、
    // 直接往 cOut 累加，因此必须把输出与输入 2（partial sum）做同 buffer inplace，
    // 否则多 K 片累加会写入未初始化的输出 buffer。
    {Opcode::OP_A_MULACC_B_EPILOGUE, {std::pair<size_t, size_t>{2, 0}}},
    {Opcode::OP_INDEX_OUTCAST, {std::pair<size_t, size_t>{2, 0}}},
};

const std::unordered_set<Opcode> inplaceOpSet = {Opcode::OP_VIEW,
                                                 Opcode::OP_ASSEMBLE,
                                                 Opcode::OP_RESHAPE,
                                                 Opcode::OP_A_MULACC_B,
                                                 Opcode::OP_A_MULACC_B_EPILOGUE,
                                                 Opcode::OP_INDEX_OUTCAST,
                                                 Opcode::OP_VIEW_TYPE};

class UnionFind {
public:
    explicit UnionFind(std::unordered_map<LogicalTensorPtr, int>& tensorToOrderIndex)
    {
        for (const auto& entry : tensorToOrderIndex) {
            const auto& tensor = entry.first;
            if (tensor == nullptr) {
                // Null operands are not valid union-find vertices.
                continue;
            }
            parentMap.emplace(tensor, tensor);
            rankMap.emplace(tensor, 1);
        }
    }

    void Unite(const LogicalTensorPtr x, const LogicalTensorPtr y)
    {
        LogicalTensorPtr xRoot = Find(x);
        LogicalTensorPtr yRoot = Find(y);
        if (xRoot == yRoot) {
            return;
        }
        // 与 Find 的未注册契约对齐: 未注册的根节点按秩 1 的独立集合补注册,
        // 避免 rankMap.at 抛出非约定的 std::out_of_range
        if (parentMap.find(xRoot) == parentMap.end()) {
            parentMap.emplace(xRoot, xRoot);
            rankMap.emplace(xRoot, 1);
        }
        if (parentMap.find(yRoot) == parentMap.end()) {
            parentMap.emplace(yRoot, yRoot);
            rankMap.emplace(yRoot, 1);
        }
        if (rankMap.at(xRoot) < rankMap.at(yRoot)) {
            parentMap[xRoot] = yRoot;
        } else if (rankMap.at(xRoot) > rankMap.at(yRoot)) {
            parentMap[yRoot] = xRoot;
        } else {
            parentMap[yRoot] = xRoot;
            rankMap.at(xRoot)++;
        }
    }

    std::vector<LogicalTensors> GetGroups()
    {
        std::unordered_map<LogicalTensorPtr, LogicalTensors> rootToGroup;
        for (const auto& pair : parentMap) {
            LogicalTensorPtr obj = pair.first;
            if (obj == nullptr) {
                // Keep any legacy null key out of the returned tensor groups.
                continue;
            }
            LogicalTensorPtr root = Find(obj);
            rootToGroup[root].push_back(obj);
        }
        std::vector<LogicalTensors> groups;
        for (const auto& pair : rootToGroup) {
            groups.push_back(pair.second);
        }
        return groups;
    }

private:
    LogicalTensorPtr Find(const LogicalTensorPtr& x)
    {
        if (x == nullptr) {
            // A null tensor has no representative in the disjoint set.
            return nullptr;
        }
        auto it = parentMap.find(x);
        if (it == parentMap.end()) {
            // 未注册的张量自成一个独立集合，以自身为代表，避免解引用尾后迭代器
            return x;
        }
        if (it->second != x) {
            it->second = Find(it->second);
        }
        return it->second;
    }
    std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> parentMap;
    std::unordered_map<LogicalTensorPtr, int> rankMap;
};

class ReplaceTensor : public Pass {
public:
    ReplaceTensor() : Pass("ReplaceTensor") {}
    ~ReplaceTensor() override = default;

private:
    /*
    补齐: Status PreCheck(Function &function) override;
    补齐: Status PostCheck(Function &function) override;
    */
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    bool CheckInplace(const Operation& op);

    std::unordered_map<LogicalTensorPtr, int> BuildTensorOrderIndexMap(Function& function);
    Status FindBaseTensor(Function& function, const std::unordered_map<LogicalTensorPtr, int>& tensorToOderIndex,
                          LogicalTensors& group, LogicalTensorPtr& baseTensor);
    void AddViewInputsToBoundTensors(const LogicalTensorPtr& curTensor, LogicalTensors& boundTensors) const;
    LogicalTensorPtr FindReplaceSource(Function& function, Operation& op,
                                       std::unordered_map<Operation*, LogicalTensorPtr>& visited);
    Status RefactorViewConnectForReplace(Function& function);
    void RefactorReplaceView(Function& function, Operation& op, const LogicalTensorPtr& srcTensor);
    void UniteTensor(Function& function, UnionFind& uf);
    void BuildRawMagicTensorIndex(Function& function);
    void SetRawTensor(const LogicalTensorPtr& tensor, const std::shared_ptr<RawTensor>& rawTensor);

    Status AlignCopyInConsumer(std::shared_ptr<LogicalTensor> tensorGm) const;
    Status AlignCopyOutProducer(std::shared_ptr<LogicalTensor> tensorGm) const;
    Status AdjustOffsetAndRawShape(LogicalTensorPtr& fromView, LogicalTensorPtr& toView) const;

    Status ForwardProcess(Function& function);
    Status BackwardProcess(Function& function);
    // 新图表达下, 当backward/forward处理完一个tensor后, 如果该tensor是assemble输出(有同rawMagic兄弟),
    // 需要同步更新兄弟的tensor指针并push到对应roots队列, 保证兄弟链路也被遍历。
    // origRawMagic < 0 时使用 tensor 当前的 rawmagic 查找兄弟;
    // >= 0 时使用指定值查找, 用于 tensor 指针已被替换但需按原始 rawMagic 定位兄弟的场景。
    void SyncSiblingAssembleOutput(Function& function, const LogicalTensorPtr& tensor, int origRawMagic = -1);

    Status ForwardView(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardAssemble(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardReshape(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardViewType(Operation* op, LogicalTensorPtr& rootTensor);
    Status ForwardCopyOut(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ForwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status ProcessForwardConsumer(Operation* op, LogicalTensorPtr& rootTensor, Function& function,
                                  std::unordered_set<int>& forwardedAssembles);

    Status BackwardAssemble(Operation* op, LogicalTensorPtr& rootTensor);
    Status BackwardReshape(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status BackwardView(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status BackwardInplaceOp(Operation* op, LogicalTensorPtr& rootTensor, Function& function);
    Status BackwardViewType(Operation* op, LogicalTensorPtr& rootTensor);
    Status BackwardInputIdx(Operation* op, LogicalTensorPtr& rootTensor, Function& function);

    Status ForUpdateView(Operation* op);
    Status BackUpdateAssemble(Operation* op);
    // L0C2UB 直写折叠：将刷新到 assemble 输入 tensor 上的目的偏移落入其唯一 producer（L0C2UB copy）
    // 的 toOffset，并将 copy 模式改为 INSERT
    Status FoldL0C2UBCopyOffset(Operation* op);
    Status RemoveFoldedAssemble(Operation& op, Operation& copyOp);
    std::vector<OpImmediate> SumOffsetForCopyIn(const std::vector<OpImmediate> offset1,
                                                const std::vector<OpImmediate> offset2);
    Status UpdateCopyInAttr(Operation* copyInOp, bool updateOffset = true);

    Status MarkTensorAsPartialMem(Function& function);

    static bool MoveSharedAssembleWriteToken(Operation& producer, Operation& consumer, const ir::VarPtr& token,
                                             Function& function);
    void RebuildTokenProducer(Function& function);
    void MoveInplaceProducerToken(Function& function, Operation& producer, const ir::VarPtr& token,
                                  const std::vector<Operation*>& matchedConsumers);
    void RebuildTokenConsumer(Function& function);

    struct CopyChainContext {
        LogicalTensorPtr input;
        LogicalTensorPtr output;
        LogicalTensorPtr ubTensor;
        Offset copyInOffset;
        bool viewOffsetAbsorbed;
    };

    Status InsertCopyUBToDDRAndDDRToUB(Function& function, Operation* op, const LogicalTensorPtr& input,
                                       const LogicalTensorPtr& output);
    Status InsertCopyDDRToUBAndUBToDDR(Function& function, Operation* op, const LogicalTensorPtr& input,
                                       const LogicalTensorPtr& output);
    Status InsertCopyDDRToUBAndUBToDDRChain(Function& function, Operation* op, const CopyChainContext& context);
    Status FindNeedToCopyAssemble(std::unordered_set<Operation*>& forwardInsertCopy,
                                  std::unordered_set<int>& visitedAssOps, Operation& op, Function& function);
    void CollectConflictingAssembleCopies(std::unordered_set<Operation*>& forwardInsertCopy,
                                          std::unordered_set<int>& visitedAssOps, Operation& op, Function& function);
    Status FindNeedToCopyReshape(std::unordered_set<Operation*>& forwardInsertCopy,
                                 std::unordered_set<int>& visitedReshapeOps, Operation& op, Function& function);
    bool HasLargerNonZeroOffsetViewProducerInput(const LogicalTensorPtr& tensor, int64_t tensorSize);
    Status FindForwardInsertCopy(Operation& op, std::unordered_set<Operation*>& forwardInsertCopy);
    int GetUbSizeThreshold();
    bool IsCopyToUBOverSize(const LogicalTensorPtr& tensor);
    Status CollectNeedInsertCopyOps(Function& function, std::unordered_set<Operation*>& forwardInsertCopy);
    Status InsertCopyForOps(Function& function, const std::unordered_set<Operation*>& ops);
    Status InsertNeedCopy(Function& function);
    bool isBoundTensor(LogicalTensorPtr& curTensor);

    bool IsAssembleSameGmWriteBack(Function& function, Operation* assembleOp) const;
    bool MatchInplaceProducerSource(const LogicalTensorPtr& assembleIn, const std::shared_ptr<RawTensor>& incastRaw,
                                    std::shared_ptr<RawTensor>& srcRaw, std::string& producerOpcode,
                                    int& producerInplaceIdx) const;

    std::unordered_map<DataType, int> viewTypeTable = {{DT_INT8, 1},    {DT_BF16, 2},    {DT_FP16, 2},   {DT_FP32, 4},
                                                       {DT_FP8E4M3, 1}, {DT_FP8E5M2, 1}, {DT_FP8E8M0, 1}};
    int ubSizeThreshold_ = -1;
    std::queue<LogicalTensorPtr> backRoots;
    std::queue<LogicalTensorPtr> forRoots;
    std::unordered_set<int> processedOp;
    std::unordered_set<int> backwardOps;
    std::unordered_set<int> forwardOps;
    // Valid only between InsertNeedCopy and the end of forward/backward traversal.
    RawMagicTensorMap rawMagicTensors_;
    IRBuilder irBuilder_;
};
} // namespace tile_fwk
} // namespace npu
#endif // REPLACE_TENSOR_H
