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
 * \file aikernel_device_task.h
 * \brief
 */

#ifndef AIKERNEL_DEVICE_TASK_H
#define AIKERNEL_DEVICE_TASK_H

#include <atomic>
#include "tilefwk/aikernel_tensor.h"
#include "tilefwk/aikernel_drco_leaf_task_ready_queue.h"
#include "tilefwk/aikernel_root_function.h"

namespace npu::tile_fwk {

struct DevStartArgsBase;

constexpr uint64_t DRCO_ROOT_FUNC_MEMORY_REUSE_STITCHED_BIT = ((uint64_t)1) << (uint64_t)60;

struct DrcoRootFuncData {
    // funcDynamicDataOffset 打包一个 u64（偏移本身平移不变）：
    /*
        |-------------------32bit------------------|-------------------32bit------------------|
        |----------succStitchListOffset------------|------------predCountOffset---------------|
        |--bit63=base标记--|------bit62-32=偏移-----|--bit31=base标记--|------bit30-0=偏移------|
        base 标记：0 = 相对 controlFlowCacheBase；1 = 相对 metadataBase
    */
    uint64_t funcDynamicDataOffset;
    // 相对 DevAscendProgram 基址的字节偏移，打包为一个 u64（偏移本身平移不变）：
    /*
        |----3bit----|-----1bit----------|------------------30bit------------------|------------------30bit------------------|
        |----free----|--dummyEndingUsed--|-------staticSuccNodeListOffset----------|------------succInfoListOffset-----------|
        staticSuccNodeListOffset：静态后继节点区（全量 DRCO 后继，分块/平铺同构）相对
        programBase 的偏移；空表 func 写 0（设备侧仅在 succNodeSize > 0 时解引用）
    */
    uint64_t funcInfoAndStaticDataOffset;
};

struct DynFuncData {
    uint64_t exprNum;              // static
    __gm__ uint64_t* opAttrs;      // static
    __gm__ int32_t* opAtrrOffsets; // static
    __gm__ uint64_t* exprTbl;      // dyn
    __gm__ DevRawTensorDesc* rawTensorDesc;
    __gm__ uint64_t* rawTensorAddr;
    uint64_t opAttrSize;
    uint64_t rawTensorDescSize;
    uint64_t rawTensorAddrSize;
    uint64_t workspaceAddr;
    __gm__ int* cceBinaryIndexList;
};

struct DynFuncBin {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
    int32_t wrapVecId{-1};
    uint8_t mixResourceType{0};
};

constexpr uint32_t MAX_AICORE_NUM_FOR_QUEUE = 108;
constexpr uint32_t NUM_LOCAL_GROUPS = (MAX_AICORE_NUM_FOR_QUEUE + LOCAL_GROUP_SIZE - 1) / LOCAL_GROUP_SIZE;
constexpr uint32_t NUM_CORE_TYPES = 2;

// 合并版 LocalMatrix：行 = 消费者（类型内本地编号，行数上限 MAX_AICORE_NUM_FOR_QUEUE，
// 覆盖 AIV 的 2*nrValidAic），每行列数上限 DRCO_MATRIX_COL_MAX、实际 colNum 运行时自适应；
// 定长 cacheline 行隔离布局（见 DrcoGlobalReadyMatrixBase）
using DrcoLocalReadyMatrix = DrcoGlobalReadyMatrixBase<LeafTaskId, MAX_AICORE_NUM_FOR_QUEUE, DRCO_MATRIX_COL_MAX>;
static_assert(alignof(DrcoLocalReadyMatrix) == CACHE_LINE_SIZE,
              "local ready matrix rows must stay cacheline aligned for u64 word exch");

struct DynFuncHeader {
    uint64_t seqNo;
    uint32_t funcNum;
    uint32_t funcSize;
    __gm__ DynFuncBin* cceBinary;
    uint64_t stackWorkSpaceAddr;
    uint64_t stackWorkSpaceSize;
    __gm__ DevStartArgsBase* startArgs;

    INLINE uint64_t GetIndex() { return seqNo; }
    INLINE uint32_t Size() { return funcNum; }
    INLINE DynFuncData& At(int index) { return (reinterpret_cast<DynFuncData*>(this + 1))[index]; }
    // 所有 DrcoRootFuncData 紧密排布在全部 DynFuncData 之后，作为只读热数据提升 cache locality
    INLINE DrcoRootFuncData* GetDrcoRootFuncDataList()
    {
        return reinterpret_cast<DrcoRootFuncData*>(reinterpret_cast<DynFuncData*>(this + 1) + funcNum);
    }
    INLINE DrcoRootFuncData& GetDrcoRootFuncData(int index) { return GetDrcoRootFuncDataList()[index]; }
};

// 取值与 CoreType::AIV/AIC/HUB_MIX 数值对齐（aikernel_data.h 中有 static_assert 强校验）：
// 设备侧解码 taskId 的 coreType 后可直接用作队列下标，无需映射
constexpr uint32_t DRCO_QUEUE_AIV = 0; // 与 CoreType::AIV 一致
constexpr uint32_t DRCO_QUEUE_AIC = 1; // 与 CoreType::AIC 一致
constexpr uint32_t DRCO_QUEUE_MIX = 2; // 与 CoreType::HUB_MIX 一致（MIX 行承载 HUB_MIX 后继）
constexpr uint32_t DRCO_QUEUE_MAX = 3;

// 全局 stitch 节点矩阵：按核类型各一个（DRCO_QUEUE_MAX 个），每个合并对应类型的
// （× group）矩阵为单一二维数组，行按核类型内本地编号索引（AIC = blockIdx，
// AIV = blockIdx - nrValidAic，行数上限 MAX_AICORE_NUM_FOR_QUEUE），列 = LOCAL_GROUP_SIZE；
// push 只写本类型核的行，pop 只读自己类型内编号对应的行
// slot（u32）编码：WAW 链 = 节点相对 stitchNodeBase 的字节偏移（< 2^31，0 = 空闲，
// FirstObjBaseOffset 保证非零）；静态后继 = bit31 静态标志 | bit30-21 生产者 funcId（10bit，
// 与 taskId funcId 字段等宽）| bit20-0 per-func 节点下标（21bit；SuccInfo 字段 16bit →
// 实际 ≤ 65535，余量充足）。消费核 pop 后按 funcId 索引 funcDataList 取节点区基址 +
// 下标还原节点（相对 funcInfoAndStaticDataOffset bit59-30）
constexpr uint32_t DRCO_STITCH_SLOT_STATIC_SUCC_FLAG = 1u << 31;
constexpr uint32_t DRCO_STITCH_SLOT_FUNCID_SHIFT = 21;
constexpr uint32_t DRCO_STITCH_SLOT_FUNCID_MASK = 0x3FFu << DRCO_STITCH_SLOT_FUNCID_SHIFT;
constexpr uint32_t DRCO_STITCH_SLOT_NODE_INDEX_MASK = 0x1FFFFFu;

INLINE uint32_t EncodeStaticSuccSlot(uint32_t funcId, uint32_t nodeIndex)
{
    return DRCO_STITCH_SLOT_STATIC_SUCC_FLAG | ((funcId & 0x3FFu) << DRCO_STITCH_SLOT_FUNCID_SHIFT) |
           (nodeIndex & DRCO_STITCH_SLOT_NODE_INDEX_MASK);
}

struct DrcoGlobalStitchNodeMatrix {
    enum {
        COL_SIZE = 1,
    };
    // 每行独占一个 64B cache line：消除跨行 false sharing（push 写相邻行不再 invalidate
    // 本行 probe）。代价：108 行 × 64B = 6.75 KB/矩阵（×3 ≈ 20 KB），可接受
    struct alignas(64) Row {
        uint32_t slot[COL_SIZE];
        uint8_t pad[64 - sizeof(uint32_t) * COL_SIZE];
    };
    Row stitchNodeList[MAX_AICORE_NUM_FOR_QUEUE];
#ifdef __TILE_FWK_HOST__
    DrcoGlobalStitchNodeMatrix()
    {
        for (uint32_t i = 0; i < MAX_AICORE_NUM_FOR_QUEUE; i++) {
            for (uint32_t j = 0; j < COL_SIZE; j++) {
                stitchNodeList[i].slot[j] = 0;
            }
        }
    }
#endif
};

// 全核共享 hub 任务矩阵：行 = 全局 blockIdx（AIC [0,nrValidAic) + AIV [nrValidAic,3*nrValidAic)），
// 承接 hubStack 溢出与 stitch 类型越界的兜底改投——任何核可 push 任意行、每核只 pop 自己行，
// fetch 循环无条件扫描（帮忙模式含），pop 后就地解依赖（不执行不计数）；own-type 就绪队列
// 因此只含可执行 leaf，批量计数无需 FIN/hub 预判
struct DrcoGlobalHubTaskMatrix {
    enum {
        COL_SIZE = 1,
    };
    // 每行独占一个 64B cache line：消除跨行 false sharing（同 DrcoGlobalStitchNodeMatrix）
    struct alignas(64) Row {
        uint32_t slot[COL_SIZE];
        uint8_t pad[64 - sizeof(uint32_t) * COL_SIZE];
    };
    Row hubTaskList[MAX_AICORE_NUM_FOR_QUEUE];
#ifdef __TILE_FWK_HOST__
    DrcoGlobalHubTaskMatrix()
    {
        for (uint32_t i = 0; i < MAX_AICORE_NUM_FOR_QUEUE; i++) {
            for (uint32_t j = 0; j < COL_SIZE; j++) {
                hubTaskList[i].slot[j] = 0;
            }
        }
    }
#endif
};

struct DrcoDevTaskFinishFlagList {
    // 每类型共享一个完成标志（原每组一个）：本 coreType 全部 leaf task 计数到 size 后置 1，
    // 本类型全部核经 DrcoGmLoad（dcci 失效读）轮询同一 flag；广播只写单个 cacheline
    struct TypeFlag {
        uint32_t devTaskFinishFlag;
        uint8_t pad[64 - sizeof(uint32_t)]; // 独占 cacheline
    };
    TypeFlag flag[DRCO_QUEUE_MAX];
#ifdef __TILE_FWK_HOST__
    DrcoDevTaskFinishFlagList()
    {
        for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ct++) {
            flag[ct].devTaskFinishFlag = 0;
        }
    }
#endif
};

// 动态 leaf 分片计数：每 coreType 一组分片（各占独立 cacheline）
constexpr uint32_t DRCO_COUNT_SHARD_NUM = 2;

// leaf task 完成计数表：承接原 DrcoGlobalReadyQueue 的 staticExecCount/size 计数职责，
// 内嵌 DrcoRootFuncList（不再独立分配）；每 coreType 一项、独占 cacheline。
// 完成判定 = Σ分片(设备侧动态原子加) >= size - staticExecCount(host 预累加的静态 per-core
// pending)：动态计数分片后各核只加自己的分片，把 64 路单点原子争抢降到 ~64/NUM 路
struct DrcoDevTaskCountList {
    struct CoreTypeCount {
        uint32_t size;            // 本 coreType leaf task 总数，host 初始化后只读
        uint32_t staticExecCount; // 静态计数：host 派发时预累加 per-core pending 数（设备侧只读）
        uint8_t pad[64 - 2 * sizeof(uint32_t)];
    };
    // 动态 leaf 计数分片：核只写自己分片，完成判定侧 dcci 读全部求和
    struct CountShard {
        uint32_t value;
        uint8_t pad[64 - sizeof(uint32_t)];
    };
    CoreTypeCount count[DRCO_QUEUE_MAX];
    CountShard shardExecCount[DRCO_QUEUE_MAX][DRCO_COUNT_SHARD_NUM];
#ifdef __TILE_FWK_HOST__
    DrcoDevTaskCountList()
    {
        for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ct++) {
            count[ct].size = 0;
            count[ct].staticExecCount = 0;
            for (uint32_t s = 0; s < DRCO_COUNT_SHARD_NUM; s++) {
                shardExecCount[ct][s].value = 0;
            }
        }
    }
#endif
};

// per-core middleQueue 槽位数：P1 单槽使用（多 region 时每槽承载一个区域队列）；可按需调整
constexpr uint32_t DRCO_MIDDLE_QUEUE_SLOT_NUM = 2;

struct DrcoRootFuncList {
    __gm__ PerCorePendingQueue* perCorePendingQueueArray[MAX_AICORE_NUM_FOR_QUEUE];
    __gm__ DrcoLocalReadyQueue* localReadyQueueArray[DRCO_QUEUE_MAX][NUM_LOCAL_GROUPS];
    // 合并版 LocalMatrix：每类型单实例（行 = 类型内本地编号，cacheline 行隔离：push 逐槽
    // u32 CAS / pop 整字 u64 exch）
    __gm__ DrcoLocalReadyMatrix* localReadyMatrixArray[DRCO_QUEUE_MAX];
    // 全核共享 stitch 节点矩阵：行 = 全局 blockIdx（AIC [0,nrValidAic) + AIV [nrValidAic,3*nrValidAic)），
    // 任何核可 push 任意行、每核只 pop 自己行——AIC/AIV 空闲侧可帮忙展开忙侧的 defer 节点
    // （节点展开类型无关：盲减 + 按 succCoreType 路由），消除分池时"一边排队一边空转"
    __gm__ DrcoGlobalStitchNodeMatrix* stitchNodeMatrix;
    // stitch pool 基址（设备地址）：stitchNodeList 槽位偏移以此为原点还原节点地址，
    // 写入于 InitDrcoRootFuncList，cache 激活时随矩阵指针一同 reloc（偏移本身平移不变）
    uint64_t stitchNodeBase;
    // 全核共享 hub 任务矩阵：hubStack 溢出/stitch 类型越界兜底任务的改投目的地，
    // fetch 循环无条件扫（帮忙模式含），slot 存 DRCO_ENCODE_TASK 的 taskId（0 = 空闲）
    __gm__ DrcoGlobalHubTaskMatrix* hubTaskMatrix;
    // control flow cache 基址（设备地址）：recording 阶段 predCount/succStitchList 落在 cache 数据区，
    // DrcoRootFuncData 的 funcDynamicDataOffset（base 标记 = 0）以此为原点还原指针
    __gm__ void* controlFlowCacheBase;
    // 元数据区基址（设备地址）：general 区起点（general + stitchPool 连续），
    // DrcoRootFuncData 的
    // funcDynamicDataOffset（base 标记 = 1）以此为原点还原指针
    __gm__ void* metadataBase;
    // DevAscendProgram 基址（设备地址）：
    // DrcoRootFuncData 的
    // funcInfoAndStaticDataOffset（bit60=dummyEndingUsed，bit59-30=staticSuccNodeListOffset，bit29-0=succInfoListOffset，
    // bit63-61 空闲）以此为原点还原指针；静态 succ slot 低 32 位偏移亦以此为原点（push/pop 同源还原）
    __gm__ void* programBase;

    alignas(64) uint32_t totalTaskCount;
    alignas(64) uint32_t devTaskFinished;
    alignas(64) DrcoDevTaskCountList devTaskCountList;
    alignas(64) DrcoDevTaskFinishFlagList devTaskFinishFlagList;
    __gm__ PerCoreMiddleQueue* perCoreMiddleQueueArray[MAX_AICORE_NUM_FOR_QUEUE][DRCO_MIDDLE_QUEUE_SLOT_NUM];
    static constexpr uint32_t DRCO_MIDDLE_QUEUE_REGION_MAX = 128;
    static constexpr uint32_t DRCO_MIDDLE_QUEUE_REGION_FLAT_MASK = 0x03FFFFFFu;
    uint32_t middleQueueRegionHeadCount{0};
    uint32_t middleQueueRegionHeadList[DRCO_MIDDLE_QUEUE_REGION_MAX];
    // 每 region 的 leaf 成员数（按核型拆分，撒核期由 host 写入）：开门核一次性代加，
    // 替代每成员核逐个原子加（16 路同计数器风暴，case3 实测 4~8us）
    uint32_t middleQueueRegionLeafCountAic[DRCO_MIDDLE_QUEUE_REGION_MAX];
    uint32_t middleQueueRegionLeafCountAiv[DRCO_MIDDLE_QUEUE_REGION_MAX];
    alignas(64) uint8_t pad[64];

    // 全局 col 游标（每 core 类型一份，该类型全部 push 核共享）：push 时按批原子认领连续
    // colIdx，colIdx -> (groupIdx = colIdx / N, col = colIdx % N)，即任务依次落到相邻消费核，
    // 跨所有组/列铺开；colIdx 达到本类型核数（groupCount * N）后回绕。追加在结构体末尾，
    // 不改变既有字段偏移。
    alignas(64) uint32_t readyMatrixPushCol[DRCO_QUEUE_MAX];

    alignas(64) uint32_t hasStitchNode;
};

#ifdef __TILE_FWK_HOST__
// per-core pending 计数预累加（设备侧 ExecDrcoPerCoreTasks 无需再原子累加），并把结果
// 备份到出参（cache 重放经 RestorePrecount 恢复，不重算）；达到 size（含 size == 0
// 空类型）即置完成 flag。仅构建路径派发完成后调用
inline void DrcoRootFuncListPrecountPerCoreTasks(DrcoRootFuncList* rootFuncList, uint32_t nrValidAic,
                                                 uint32_t* executedBackup, uint32_t* finishFlagBackup)
{
    for (uint32_t i = 0; i < MAX_AICORE_NUM_FOR_QUEUE; i++) {
        PerCorePendingQueue* queue = rootFuncList->perCorePendingQueueArray[i];
        if (queue == nullptr || queue->size == 0) {
            continue;
        }
        // 核归属与设备侧 DrcoGetCoreTypedIdx 一致：编号 < nrValidAic 为 AIC，其余为 AIV
        uint32_t coreType = (i < nrValidAic) ? DRCO_QUEUE_AIC : DRCO_QUEUE_AIV;
        rootFuncList->devTaskCountList.count[coreType].staticExecCount += queue->size;
    }
    for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ct++) {
        DrcoDevTaskCountList::CoreTypeCount* cnt = &rootFuncList->devTaskCountList.count[ct];
        executedBackup[ct] = cnt->staticExecCount;
        if (cnt->staticExecCount >= cnt->size) {
            rootFuncList->devTaskFinishFlagList.flag[ct].devTaskFinishFlag = 1;
        }
        finishFlagBackup[ct] = rootFuncList->devTaskFinishFlagList.flag[ct].devTaskFinishFlag;
    }
}

// cache 重放侧恢复构建期备份的 precount；须在 DrcoReadyQueueDataRestore 的计数复位之后
inline void DrcoRootFuncListRestorePrecount(DrcoRootFuncList* rootFuncList, const uint32_t* staticExecCount,
                                            const uint32_t* finishFlag)
{
    for (uint32_t ct = 0; ct < DRCO_QUEUE_MAX; ct++) {
        rootFuncList->devTaskCountList.count[ct].staticExecCount = staticExecCount[ct];
        rootFuncList->devTaskFinishFlagList.flag[ct].devTaskFinishFlag = finishFlag[ct];
    }
}
#endif

constexpr int32_t DEVICE_TASK_QUEUE_SIZE = 64;
struct DrcoDeviceTask {
    __gm__ DynFuncHeader* dynFuncDataList;
    __gm__ DrcoRootFuncList* drcoRootFuncList;
};
struct DrcoDeviceTaskReadyQueue {
    uint32_t head;
    uint32_t tail;
    DrcoDeviceTask dynFuncDataListList[DEVICE_TASK_QUEUE_SIZE];
#ifdef __TILE_FWK_HOST__
    void Reset()
    {
        head = 0;
        tail = 0;
    }
    bool TryAppend(DynFuncHeader* dynFuncDataList, DrcoRootFuncList* rootFuncList)
    {
        if (tail - __atomic_load_n(&head, __ATOMIC_ACQUIRE) == DEVICE_TASK_QUEUE_SIZE) {
            return false;
        }
        uint32_t idx = tail % DEVICE_TASK_QUEUE_SIZE;
        dynFuncDataListList[idx].dynFuncDataList = dynFuncDataList;
        dynFuncDataListList[idx].drcoRootFuncList = rootFuncList;
        __sync_synchronize();
        __atomic_store_n(&tail, tail + 1, __ATOMIC_RELEASE);
        return true;
    }
#endif
};

} // namespace npu::tile_fwk

#endif
