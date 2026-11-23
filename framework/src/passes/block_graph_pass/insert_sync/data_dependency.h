/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for details.
 */
#ifndef PASS_INSERT_SYNC_DEPENDENCY_H
#define PASS_INSERT_SYNC_DEPENDENCY_H
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include "interface/operation/operation.h"
namespace npu {
namespace tile_fwk {
struct Interval {
    int start;
    int end;
    int idx;
    Interval(int s, int e, int id) : start(s), end(e), idx(id) {}
};

struct IntervalTreeNode {
    Interval interval;
    int max;
    IntervalTreeNode* left;
    IntervalTreeNode* right;
    explicit IntervalTreeNode(Interval i) : interval(i), max(i.end), left(nullptr), right(nullptr) {}
};

class RangeSearchTree {
public:
    ~RangeSearchTree() { FreeTree(); }
    void Insert(int left, int right, int idx);
    std::set<int> GetCovered(int left, int right);

private:
    void FreeTree();
    Status ProcessTreeNode(const Interval& interval, IntervalTreeNode* currPtr,
                           std::vector<IntervalTreeNode*>& intervalStack);
    Status InsertInterval(const Interval& interval);
    void OverlapSearch(const Interval& interval, std::set<int>& result);
    IntervalTreeNode* treeRoot = nullptr;
};

class DataDependencySearcher {
public:
    std::set<int> Find(Operation* opWait);
    void Insert(const Operation* opSet, int idx);
    std::unordered_map<int, TileRange> ubTensorRangeMap;

private:
    void CheckRAWSearchTree(Operation* opWait, std::set<int>& res);
    void CheckWAWSearchTree(Operation* opWait, std::set<int>& res);
    void CheckWARSearchTree(Operation* opWait, std::set<int>& res);
    void InsertWAWSearchTree(const Operation* opSet, int idx);
    void InsertRAWSearchTree(const Operation* opSet, int idx);
    void InsertWARSearchTree(const Operation* opSet, int idx);

    std::unordered_map<MemoryType, RangeSearchTree> wawSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> warSearchTree_;
    std::unordered_map<MemoryType, RangeSearchTree> rawSearchTree_;
    std::unordered_map<int, std::set<int>> readDdrMemMap;
    std::unordered_map<int, std::set<int>> writeDdrMemMap;
    std::unordered_map<const Operation*, int> opIndices_;
};

} // namespace tile_fwk
} // namespace npu
#endif
