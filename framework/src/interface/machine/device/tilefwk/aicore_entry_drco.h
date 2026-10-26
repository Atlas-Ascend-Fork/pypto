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
 * \file aicore_state_drco.h
 * \brief DRCO (Dependency Resolving by aiCOre) scheduling flow for aicore_state.h.
 */

#ifndef AICORE_ENTRY_DRCO_H
#define AICORE_ENTRY_DRCO_H

#include "tilefwk/aicore_pto_debug.h"

#ifdef __DAV_C310__

#if defined(__MIX__)
#define IS_MIX 1
#else
#define IS_MIX 0
#endif

#if defined(__AIV__)
#define IS_AIV 1
#else
#define IS_AIV 0
#endif

#if defined(__AIC__)
#define IS_AIC 1
#else
#define IS_AIC 0
#endif

namespace npu::tile_fwk {

#if defined(__AIV__)
constexpr uint32_t DRCO_CORE_TYPE = static_cast<uint32_t>(CoreType::AIV);
#elif defined(__AIC__)
constexpr uint32_t DRCO_CORE_TYPE = static_cast<uint32_t>(CoreType::AIC);
#endif

} // namespace npu::tile_fwk

using npu::tile_fwk::DRCO_CORE_TYPE;
using npu::tile_fwk::LOCAL_GROUP_SIZE;

using npu::tile_fwk::FuncID;
using npu::tile_fwk::TaskID;

#define DRCO_DCCI_SINGLE_CACHE_LINE(ptr) dcci((__gm__ uint8_t*)ptr, SINGLE_CACHE_LINE, CACHELINE_OUT)
#define DRCO_DCCI_ENTIRE_DATA_CACHE() dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT)

#define DRCO_BUSY_BACKOFF_CYC 500
// 共享 stitch 矩阵的全核行域倍率：AIC [0,nrValidAic) + AIV [nrValidAic, 3*nrValidAic)，
// 即全核数 = AIC 1 倍 + AIV 2 倍；矩阵行 = 全局 blockIdx，行数上限钳到 MAX_AICORE_NUM_FOR_QUEUE
constexpr uint32_t DRCO_ALL_CORES_PER_AIC = 3;

#if ENABLE_AICORE_PRINT
#define DRCO_LOG(ctx, fmt, ...) AICORE_LOGE((ctx)->logger.Context(), fmt, ##__VA_ARGS__)
#else
#define DRCO_LOG(ctx, fmt, ...) \
    do {                        \
    } while (0)
#endif

INLINE void DrcoBusyBackOff()
{
    uint64_t t0 = get_sys_cnt();
    while (get_sys_cnt() - t0 < DRCO_BUSY_BACKOFF_CYC) {
    }
}

#define DRCO_LEADER_TIMEOUT_CHECK(t0, loopCount, timelen, lastStatus) \
    ++loopCount;                                                      \
    if ((loopCount % 1000 == 0)) {                                    \
        uint64_t elapsed = get_sys_cnt() - t0;                        \
        if (!warningSet && elapsed > AICORE_WARNING_CYCLES) {         \
            SetWarningStatus(state->args, lastStatus);                \
            warningSet = true;                                        \
        }                                                             \
        if (elapsed > (timelen)) {                                    \
            SetLastWordStatus(state->args, lastStatus);               \
            SyncAllMix();                                             \
            Trap();                                                   \
            return nullptr;                                           \
        }                                                             \
    }

constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;
constexpr uint32_t BATCH_PUSH_BUF_SIZE = 6;

__aicore__ inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_FLAG_ID_MAX = 16;

__aicore__ inline void SyncAllMix()
{
#if IS_AICORE
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0, SYNC_AIC_FLAG));
    wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX);
#elif defined(__DAV_VEC__)
    set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

template <typename T>
INLINE T DrcoGmLoad(__gm__ T* ptr)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(ptr);
    return *ptr;
}

template <typename T>
INLINE void DrcoGmStore(__gm__ T* ptr, T value)
{
    *ptr = value;
    DRCO_DCCI_SINGLE_CACHE_LINE(ptr);
}

template <typename T>
INLINE T DrcoGmLoadArray(__gm__ T* ptr, uint32_t idx)
{
    DRCO_DCCI_SINGLE_CACHE_LINE(&ptr[idx]);
    return ptr[idx];
}

template <typename T>
INLINE void DrcoGmStoreArray(__gm__ T* ptr, uint32_t idx, T value)
{
    ptr[idx] = value;
    DRCO_DCCI_SINGLE_CACHE_LINE(&ptr[idx]);
}

INLINE uint32_t DrcoAtomicLoad(__gm__ uint32_t* ptr) { return static_cast<uint32_t>(atomicCAS(ptr, 0, 0)); }

INLINE uint32_t DrcoAtomicAddToS32(__gm__ int32_t* ptr, int32_t value)
{
    return static_cast<uint32_t>(atomicAdd(ptr, value));
}

INLINE uint32_t DrcoAtomicAddToU32(__gm__ uint32_t* ptr, uint32_t value)
{
    return static_cast<uint32_t>(atomicAdd(ptr, value));
}

INLINE uint32_t DrcoAtomicCasToU32(__gm__ uint32_t* ptr, uint32_t compare, uint32_t value)
{
    return static_cast<uint32_t>(atomicCAS(ptr, compare, value));
}

INLINE uint32_t DrcoAtomicExchToU32(__gm__ uint32_t* ptr, uint32_t value)
{
    return static_cast<uint32_t>(atomicExch(ptr, value));
}

INLINE uint64_t DrcoAtomicExchToU64(__gm__ uint64_t* ptr, uint64_t value)
{
    return static_cast<uint64_t>(atomicExch(ptr, value));
}

using DrcoDeviceTask = npu::tile_fwk::DrcoDeviceTask;
using DrcoDeviceTaskReadyQueue = npu::tile_fwk::DrcoDeviceTaskReadyQueue;
using DrcoLocalReadyQueue = npu::tile_fwk::DrcoLocalReadyQueue;
using DrcoLocalReadyMatrix = npu::tile_fwk::DrcoLocalReadyMatrix;
using DrcoGlobalStitchNodeMatrix = npu::tile_fwk::DrcoGlobalStitchNodeMatrix;
using DevAscendFunctionOperationSuccInfo = npu::tile_fwk::DevAscendFunctionOperationSuccInfo;
using DevAscendFunctionDuppedStitchNode = npu::tile_fwk::DevAscendFunctionDuppedStitchNode;

// 不确定bisheng在处理小于 64bit 的结构体的时候，是否能够完全保存在寄存器中，先用 uint64_t + bit 计算的方式处理
typedef uint64_t BlockDesc;

template <uint64_t lo, uint64_t hi>
INLINE uint64_t BlockDescInsert(BlockDesc data)
{
    return (data & (((uint64_t)1 << (hi - lo)) - 1)) << lo;
}
template <uint64_t lo, uint64_t hi>
INLINE uint64_t BlockDescExtract(BlockDesc data)
{
    return (data >> lo) & (((uint64_t)1 << (hi - lo)) - 1);
}
INLINE uint64_t BlockDescExtract(BlockDesc data, uint64_t lo, uint64_t hi)
{
    return (data >> lo) & (((uint64_t)1 << (hi - lo)) - 1);
}

INLINE BlockDesc BlockDescCreate(uint64_t blockIdx, uint64_t typedBlockIdx, uint64_t validAicNum, uint64_t validAivNum)
{
    return BlockDescInsert<24, 32>(blockIdx) | BlockDescInsert<16, 24>(typedBlockIdx) |
           BlockDescInsert<8, 16>(validAivNum) | BlockDescInsert<0, 8>(validAicNum);
}
INLINE uint32_t BlockDescBlockIdx(BlockDesc desc) { return BlockDescExtract<24, 32>(desc); }
INLINE uint32_t BlockDescTypedBlockIdx(BlockDesc desc) { return BlockDescExtract<16, 24>(desc); }
INLINE uint32_t BlockDescValidCoreNum(BlockDesc desc, bool isAiv)
{
    return BlockDescExtract(desc, 0 + isAiv * 8, 8 + isAiv * 8);
}

#define ENABLE_AICORE_TRACE 0

#ifndef DRCO_FLAG_HAS_MIX
#define DRCO_FLAG_HAS_MIX 0
#endif
// 直通 selfReadyQueue 容量：同类型就绪后继截留直通上限（resolve 收尾直写本核槽位，
// fetch 循环每轮开头直取，省 push/pop 矩阵探测往返；波内实测容量 1 即饱和）
constexpr uint32_t SELF_QUEUE_SIZE = 1;

// fetch 批取大小（与 LOCAL_GROUP_SIZE/矩阵维度解耦）：批量 pop 上限。实测波内单行
// 多数时刻可取任务数 < 矩阵维度，批上限过大只是放宽并发上限并不加速命中——取小值
// 减小批内串行执行尾长，让任务更快回到调度（取到即走），也缩短 deferred 批尾 resolve 链
constexpr uint32_t DRCO_FETCH_BATCH_SIZE = 4;

// hub 级联栈容量：必须 ≥ 初始批填入上限（DrcoResolveDepend 把整批 fetched 任务压栈，
// 上限即 DRCO_FETCH_BATCH_SIZE）——LGS8 后按 4 取批而栈仍为 2 会越界写栈帧（507015 实证）；
// hub 链级联超深走 hub matrix 改投（DrcoResolveDependOnce 溢出分支），容量只影响矩阵
// 流量不影响正确性，取 fetch batch + 级联余量
constexpr uint32_t HUB_STACK_SIZE = DRCO_FETCH_BATCH_SIZE + 4;

struct DrcoEntryState {
    BlockDesc blockDesc;
    __gm__ KernelArgs* args;
    __gm__ FedProfPerCoreProf* metric;
    ExecuteContext ctx;
    __gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData* runtimeDataRingBufferHeadData;
    __gm__ npu::tile_fwk::DevStartArgsBase* base;
    __gm__ DrcoDeviceTaskReadyQueue* deviceTaskReadyQueue;
    uint8_t lastMixResourceType;

    uint32_t selfReadyQueueSize;
    uint32_t selfReadyQueue[SELF_QUEUE_SIZE];
    // SSBuf 段（__ssbuf__ 专用地址空间，段基址 0 合法）上的 MIX C2V 通信状态
    npu::tile_fwk::DrcoSsbufState* ssbuf;
    npu::tile_fwk::DrcoMixHubC2VReadyQueue* mixHubC2VReadyQueue;
    bool isExectedLeafTask{false};
    // middleQueue 全量核本地状态（每 devTask 复位/载入一次，热路径只读）：
    //  · queue：槽队列指针（从 rootFuncList 载入；null = 无直通场景）
    //  · headTaskId：门探测（encode 期不变量，热路径只读其 predCount，1 次/轮）
    //  · queueSize：槽深度缓存（与 headTaskId 同 cacheline 一次 u64 读入，consume 免冷 dcci）
    //  · consumed：本轮消费标记（队列内容跨回放只读）
    //  · pending：门已开标记（fetch 自旋内置位，重消费在外层执行）
    //  · verifiedSlot：probe 已验开的槽位号（0xFF = 无），consume 对该槽跳过门复读——
    //    16 核同时复读同一 head predCount cacheline 是开门后最大争用点（case3 实测 ~2us）；
    //    probe 刚做过新鲜 DCCI 读，结果直接复用（0xff 粘性 + consumed 防重入，跳过安全）
    //  · regionHeadCount/regionHeadList：区域头注册表核本地副本（TryOpen 完成点高频比对，
    //    每 devTask 载入一次，避免读 GM）
    struct MiddleQueueState {
        __gm__ npu::tile_fwk::PerCoreMiddleQueue* queue[npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM];
        uint32_t headTaskId[npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM];
        uint32_t queueSize[npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM];
        bool consumed[npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM];
        bool pending;
        uint8_t verifiedSlot{0xFF};
        uint32_t regionHeadCount;
        uint32_t regionHeadList[npu::tile_fwk::DrcoRootFuncList::DRCO_MIDDLE_QUEUE_REGION_MAX];
    } middleQueue;
    // 期望动态计数 = 本 devTask 本 coreType 的 leaf 总数(size) - host 静态预累加计数(staticExecCount)
    // （devTask 内不变）：入口缓存一次，热路径完成判定用 Σ分片 >= 该值；动态计数走分片，
    // 避免对热 staticExecCount/size 行反复 dcci
    uint32_t drcoCountDynExpect{0};
    // 本 devTask 是否存在 stitch 节点（devTask 内不变）：入口缓存一次；为 false 时 stitch
    // 矩阵恒无生产者，fetch 循环整段跳过其扫描，省每轮 dcci 空探
    bool hasStitchNode{false};
    // C 端聚合状态核本地缓存（if (IS_AIC) 运行时门控，散落字段整合）：
    //  · vectorCountDynExpect：AIV 期望动态计数（devTask 内不变，入口缓存）
    //  · lastVCount：本组 V0/V1 计数快照
    //  · vectorFinishPublished：FINISH_VECTOR 发布标记
    //  · vectorFinishPollTick：完成判定轮询计数（降频兜底，每 devTask 复位保证首轮即判）
    struct WarpAggState {
        uint32_t vectorCountDynExpect{0};
        uint32_t lastVCount[2]{0, 0};
        bool vectorFinishPublished{false};
        uint32_t vectorFinishPollTick{0};
    } warpAggState;

    // control flow cache 基址（设备地址）：recording 阶段 predCount/succStitchList 落在 cache 数据区，
    // funcDynamicDataOffset（base 标记 = 0）以此为原点还原指针
    __gm__ void* controlFlowCacheBase;
    // 元数据区基址（设备地址）：general 区起点（general + stitchPool 连续），
    // funcDynamicDataOffset（base 标记 = 1）以此为原点还原指针
    __gm__ void* metadataBase;
    // DevAscendProgram 基址（设备地址）：
    // DrcoRootFuncData 的
    // funcInfoAndStaticDataOffset（bit60=dummyEndingUsed，bit59-30=staticSuccNodeListOffset，bit29-0=succInfoListOffset，
    // bit63-61 空闲）以此为原点还原指针
    __gm__ void* programBase;

#if ENABLE_AICORE_TRACE
    struct TraceEventStatistic {
        uint32_t taskIndex;
        struct TraceEvent {
            uint64_t timestamp;
            uint32_t taskId;

#define EVENT_SIZE 0x400
#define EVENT(hi, lo) (((hi) << 16) | ((lo) & 0xffff))
#define EVENT_LEAF_START() EVENT(1, 0x1001)
#define EVENT_LEAF_END() EVENT(2, 0x1002)
#define EVENT_RESOLVE() EVENT(3, 0)
#define EVENT_ROWCURSOR() EVENT(3, 0x103)
#define EVENT_PUSH_LOCAL(group) EVENT(4, group)
#define EVENT_PUSH_GLOBAL() EVENT(5, 0)
#define EVENT_SUCC_STATIC(size) EVENT(6, size)
#define EVENT_SUCC_DYNAMIC(size) EVENT(7, size)
#define EVENT_STITCH_NODE(size) EVENT(8, size)
#define EVENT_FETCH_START() EVENT(9, 0x1000)
#define EVENT_FETCH_END() EVENT(9, 0x2000)
#define EVENT_PUSH_SELF_READY_QUEUE() EVENT(10, 0x1000)
#define EVENT_POP_SELF_READY_QUEUE() EVENT(11, 0x1000)
            uint32_t eventCode;
        } traceEventList[EVENT_SIZE];
    } traceEventStatistic;
#endif
};
#define FUNCID_TASKID(id) FuncID(id), TaskID(id)

#if ENABLE_AICORE_TRACE
INLINE static void _TraceInit(DrcoEntryState* state) { state->traceEventStatistic.taskIndex = 0; }

INLINE static void _TraceEvent(DrcoEntryState* state, uint32_t taskId, uint32_t eventCode)
{
    if (state->traceEventStatistic.taskIndex < EVENT_SIZE) {
        auto* code = &state->traceEventStatistic.traceEventList[state->traceEventStatistic.taskIndex++];
        code->timestamp = get_sys_cnt();
        code->taskId = taskId;
        code->eventCode = eventCode;
    }
}

INLINE static void _TracePrint(DrcoEntryState* state)
{
    auto traceEventStatistic = &state->traceEventStatistic;
    PYPTO_AICORE_PRINTF("total=%d", traceEventStatistic->taskIndex);
    for (uint32_t i = 0; i < traceEventStatistic->taskIndex; i++) {
        auto event = &traceEventStatistic->traceEventList[i];
        (void)event;
        PYPTO_AICORE_PRINTF("timestamp=%llu, code=%x taskId=%d:%d-%x duration=%llu",
                            (unsigned long long)event->timestamp, event->eventCode, FUNCID_TASKID(event->taskId),
                            event->taskId >> npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_SHIFT,
                            i == 0 ? 0 : (event->timestamp - traceEventStatistic->traceEventList[i - 1].timestamp));
    }
}
#define TraceInit(state) _TraceInit(state)
#define TraceEvent(state, taskId, eventCode) _TraceEvent(state, taskId, eventCode)
#define TracePrint(state) _TracePrint(state)
#else
#define TraceInit(state)
#define TraceEvent(state, taskId, eventCode)
#define TracePrint(state)
#endif

// predCountOffset / succStitchListOffset 最高位（bit31）为 base 标记：
// 0 = 相对 controlFlowCacheBase；1 = 相对 metadataBase。低 31 位为实际字节偏移。
INLINE __gm__ int32_t* DrcoGetCurrentRootFuncPredCount(__gm__ void* controlFlowCacheBase, __gm__ void* metadataBase,
                                                       uint32_t predCountOffset)
{
    __gm__ void* base = (predCountOffset & 0x80000000u) ? metadataBase : controlFlowCacheBase;
    return reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<uint64_t>(base) + (predCountOffset & 0x7fffffffu));
}

INLINE __gm__ DevAscendFunctionDuppedStitchNode** DrcoGetCurrentRootFuncSuccStitchList(
    __gm__ void* controlFlowCacheBase, __gm__ void* metadataBase, uint32_t succStitchListOffset)
{
    __gm__ void* base = (succStitchListOffset & 0x80000000u) ? metadataBase : controlFlowCacheBase;
    return reinterpret_cast<__gm__ DevAscendFunctionDuppedStitchNode**>(reinterpret_cast<uint64_t>(base) +
                                                                        (succStitchListOffset & 0x7fffffffu));
}

// 静态后继节点区基址（相对 programBase，偏移打包于 funcInfoAndStaticDataOffset bit59-30）：
// 全量 DRCO 后继统一以节点存储；空表 func 偏移为 0——仅在 succNodeSize > 0 时解引用
INLINE __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* DrcoGetCurrentRootFuncStaticSuccNodeList(
    __gm__ void* programBase, uint64_t funcInfoAndStaticDataOffset)

{
    uint32_t nodeRegionOffset = static_cast<uint32_t>((funcInfoAndStaticDataOffset >> 30) & 0x3FFFFFFFULL);
    return reinterpret_cast<__gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode*>(
        reinterpret_cast<uint64_t>(programBase) + nodeRegionOffset);
}
INLINE __gm__ npu::tile_fwk::DrcoRootFuncData* DrcoGetCurrentRootFuncData(DrcoEntryState* state, uint32_t funcIdx)
{
    return &state->ctx.cachedDevTaskCurr->drcoRootFuncDataList[funcIdx];
}
// taskId 的 predCount GM 地址（middleQueue 门协议）
// special 位在 FuncID/TaskID 实际位宽之上，解码天然屏蔽
INLINE __gm__ int32_t* DrcoTaskPredCountAddr(DrcoEntryState* state, uint32_t taskId)
{
    __gm__ npu::tile_fwk::DrcoRootFuncData* rootFuncData = DrcoGetCurrentRootFuncData(state,
                                                                                      npu::tile_fwk::FuncID(taskId));
    return DrcoGetCurrentRootFuncPredCount(state->controlFlowCacheBase, state->metadataBase,
                                           static_cast<uint32_t>(rootFuncData->funcDynamicDataOffset)) +
           npu::tile_fwk::TaskID(taskId);
}

// 区域头开门：按 flatId（func+op，bits0-25）比对核本地注册表副本（一解多直通头恒满覆盖，
// 命中即头——写头 predCount=0xff 开门，调用方跳过 resolve；未命中返回 false 走常规路径）。
// 开门同时由本核一次性代加该 region 的 leaf 总数（按核型拆账，host 撒核期写入）——替代
// 原消费侧每核逐个原子加（16 路同计数器风暴，case3 实测 4~8us）。写序恒为 0xff → add →
// flag：退出路径据此做"flag 置位 ⇒ 门已全开"的双检（见 fetch 循环退出分支）
INLINE void DrcoNotifyTaskExecutedAddType(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                          uint32_t queueType, uint32_t addCount);
INLINE bool DrcoMiddleQueueRegionHeadTryOpen(DrcoEntryState* state,
                                             __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t taskId)
{
    const uint32_t count = state->middleQueue.regionHeadCount;
    if (count == 0) {
        return false;
    }
    const uint32_t flat = taskId & npu::tile_fwk::DrcoRootFuncList::DRCO_MIDDLE_QUEUE_REGION_FLAT_MASK;
    for (uint32_t i = 0; i < count; i++) {
        if (state->middleQueue.regionHeadList[i] == flat) {
            // 开门必须字粒度原子交换：DrcoGmStore 的 dcci(CACHELINE_OUT) 会把本核 L1 中
            // 过期的整行 predCount（头与邻居 op 共享 cacheline）回写 GM，抹掉其他核对同行
            // 邻居的原子减计数（mla_prolog capture 实证：头 op394/395 与 op393 同行，开门
            // 回写使 op393 的前序减计数回退 → 永不点火 → 下游 8 任务不就绪 → fetch 挂死）。
            // 原子交换走 L2 字粒度，不回写 cacheline。
            DrcoAtomicExchToU32(reinterpret_cast<__gm__ uint32_t*>(DrcoTaskPredCountAddr(state, taskId)),
                                npu::tile_fwk::DRCO_MIDDLE_QUEUE_GATE_OPEN);
            // 开门核本地已知门开：按 flat 匹配直置 verifiedSlot，consume 跳过门复读
            // （head predCount 行是 16 核轮询热线，开门核未走 probe 无缓存，复读最贵）
            for (uint32_t s = 0; s < npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM; s++) {
                if ((state->middleQueue.headTaskId[s] &
                     npu::tile_fwk::DrcoRootFuncList::DRCO_MIDDLE_QUEUE_REGION_FLAT_MASK) == flat) {
                    state->middleQueue.verifiedSlot = static_cast<uint8_t>(s);
                    break;
                }
            }
            DrcoNotifyTaskExecutedAddType(state, rootFuncList, npu::tile_fwk::DRCO_QUEUE_AIC,
                                          DrcoGmLoad(&rootFuncList->middleQueueRegionLeafCountAic[i]));
            DrcoNotifyTaskExecutedAddType(state, rootFuncList, npu::tile_fwk::DRCO_QUEUE_AIV,
                                          DrcoGmLoad(&rootFuncList->middleQueueRegionLeafCountAiv[i]));
            return true;
        }
    }
    return false;
}

// 门判定 forward：fetch 自旋内使用，完整定义见 ExecDrcoPerCoreTasks 之前
INLINE bool DrcoMiddleQueueGateOpen(DrcoEntryState* state, uint32_t s);

// middleQueue 门探测（fetch 自旋内逐轮调用）：任一未消费槽的门开（头 predCount==0xff）
// → 置 pending、置空批并返回 true，调用方直接 break 退出自旋，重消费在外层执行
INLINE bool DrcoMiddleQueueTryProbeGate(DrcoEntryState* state, uint32_t& resultTaskIdCount)
{
    for (uint32_t s = 0; s < npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM; s++) {
        if (state->middleQueue.consumed[s] || state->middleQueue.queue[s] == nullptr) {
            continue;
        }
        if (DrcoMiddleQueueGateOpen(state, s)) {
            state->middleQueue.pending = true;
            state->middleQueue.verifiedSlot = static_cast<uint8_t>(s);
            resultTaskIdCount = 0;
            return true;
        }
    }
    return false;
}

INLINE __gm__ DevAscendFunctionOperationSuccInfo* DrcoGetCurrentRootFuncSuccInfoList(__gm__ void* programBase,
                                                                                     uint32_t succInfoListOffset)
{
    return reinterpret_cast<__gm__ DevAscendFunctionOperationSuccInfo*>(((uint64_t)programBase) + succInfoListOffset);
}

INLINE __gm__ DrcoDeviceTask* GetCurrentDeviceTask(__gm__ DrcoDeviceTaskReadyQueue* queue)
{
    uint32_t head = DrcoGmLoad(&queue->head);
    uint32_t tail = DrcoGmLoad(&queue->tail);
    if (head < tail) {
        __gm__ DrcoDeviceTask* elem = &queue->dynFuncDataListList[head % npu::tile_fwk::DEVICE_TASK_QUEUE_SIZE];
        if (DrcoGmLoad(&elem->dynFuncDataList) != nullptr) {
            return elem;
        }
    }
    return nullptr;
}

template <typename T, T SENTINEL, T DEC>
INLINE T DrcoAtomicResolveDependOnce(__gm__ T* ptr)
{
    if (*ptr == SENTINEL) {
        return SENTINEL;
    }
    return static_cast<T>(atomicAdd(ptr, DEC));
}

// ==================== LocalReadyMatrix 原语：push 逐槽 u32 CAS 写 / pop 整字 u64 exch 读清 ====================

// 类型内本地编号：blockIdx < nrValidAic（AIC 核）为 blockIdx，否则（AIV 核）为 blockIdx - nrValidAic。
// LocalQueue 仍按分组索引（group = 本地编号 / LOCAL_GROUP_SIZE）；
// LocalMatrix 合并为每类型单实例，行 = 本地编号本身
INLINE uint32_t DrcoGetCoreTypedIdx(uint32_t blockIdx, uint32_t nrValidAic)
{
    return blockIdx < nrValidAic ? blockIdx : blockIdx - nrValidAic;
}

// 消费者行 row（行 = 类型内本地编号）的第 wordIdx 个 u64 字：行内槽连续，
// 行 64B 对齐保证字 8B 对齐
INLINE __gm__ uint64_t* DrcoLocalReadyMatrixWord(__gm__ DrcoLocalReadyMatrix* matrix, uint32_t wordIdx, uint32_t row)
{
    return reinterpret_cast<__gm__ uint64_t*>(&matrix->rowList[row].taskList[wordIdx * 2]);
}

INLINE __gm__ DrcoLocalReadyMatrix* DrcoRootFuncListGetLocalReadyMatrix(
    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t readyMatrixCoreType);

// 遍历深度环 × 消费者行环：任务行按全局游标 readyMatrixPushCol[queueType] 整批原子认领
// 轮转分配（base = 认领起点 % rowCnt，批内相邻任务落相邻消费者核，跨所有组/列铺开）；
// 起始深度列 = push 核身份散列（typedBlockIdx % colNum），当前列槽被占则沿深度环换下一
// 列重试，本行 colNum 个深度全满才从本任务起（list[pushed..]）整体溢出去队列；
// 逐槽 u32 CAS（0→任务）一槽一原子、槽间独立，pop 侧按包含该槽的
// u64 整字 exch 读清——跨位宽 RMW 在 L2 按地址串行化，互不丢任务；
// validCoreNum == 0 禁写由调用方早退保证；每个 CAS 成功的任务记录 EVENT_PUSH_LOCAL(消费者) 事件。
INLINE uint32_t DrcoLocalReadyMatrixPushBatch(DrcoEntryState* state,
                                              __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                              uint32_t typedBlockIdx, uint32_t* succTaskIdList,
                                              uint32_t succTaskIdListSize, uint32_t queueType)
{
    __gm__ DrcoLocalReadyMatrix* matrix = DrcoRootFuncListGetLocalReadyMatrix(rootFuncList, queueType);
    if (matrix == nullptr || matrix->validCoreNum == 0) {
        return 0;
    }
    // 当前的matrix 有多少行
    uint32_t colNum = matrix->colNum;
    uint32_t rowCnt = matrix->validCoreNum;
    TraceEvent(state, succTaskIdList[0], EVENT_ROWCURSOR());
    __gm__ uint32_t* rowCursor = &rootFuncList->readyMatrixPushCol[queueType];
    // succTaskIdListSize <=6 不会溢出；即便回绕，% coreCount 仍给合法核
    uint32_t base = DrcoAtomicAddToU32(rowCursor, succTaskIdListSize) % rowCnt;
    // 起始深度列 = 推送核身份散列；该槽被占则沿深度环换下一列重试
    uint32_t matrixColStart = typedBlockIdx % colNum;
    uint32_t pushed = 0;
    for (uint32_t i = 0; i < succTaskIdListSize; i++) {
        uint32_t rowIdx = base + i;
        if (rowIdx >= rowCnt) {
            rowIdx -= rowCnt;
        }
        // 依据 行idx 计算在本次matrix 中行 列
        uint32_t matrixRowIdx = rowIdx % rowCnt;
        uint32_t matrixColIdx = matrixColStart;
        bool slotPushed = false;
        for (uint32_t k = 0; k < colNum; k++) {
            if (DrcoAtomicCasToU32(&matrix->rowList[matrixRowIdx].taskList[matrixColIdx], 0,
                                   DRCO_ENCODE_TASK(succTaskIdList[i])) == 0) {
                slotPushed = true;
                TraceEvent(state, succTaskIdList[i], EVENT_PUSH_LOCAL(rowIdx));
                break;
            }
            if (++matrixColIdx == colNum) {
                matrixColIdx = 0; // 深度环回绕
            }
        }
        if (!slotPushed) {
            // 本行 colNum 个深度全占（消费侧未及时排空）：从本任务起（list[pushed..]）整体溢出去队列
            break;
        }

        pushed++;
    }
    return pushed;
}

// 消费者本行（行内槽连续）整字 u64 exch 读清：一次原子取相邻 2 任务，读清一体即独占
// （安全性：每行仅本行主一个 popper、push 侧只 CAS 空槽，读清无需二次校验）；
// maxCount 限制本次最多取出的任务数，整字粒度取数可能超出 1 个（最后一个字贡献 2 个
// 任务），调用方数组按 DRCO_MATRIX_COL_MAX 定尺寸即可覆盖；无游标——字即状态，
INLINE uint32_t DrcoLocalReadyMatrixPopRowTasks(__gm__ DrcoLocalReadyMatrix* matrix, uint32_t rowIdx,
                                                uint32_t* outTaskList, uint32_t maxCount)
{
    uint32_t wordCnt = matrix->colNum >> 1;
    uint32_t count = 0;
    // 行恒为单个 64B cacheline（Row alignas(64)，taskList ≤ DRCO_MATRIX_COL_MAX=8 槽 = 32B）：
    // 一次 dcci 失效整行即可，随后逐字普通读命中 L1；原逐字 DrcoGmLoad 对同一行重复 dcci，
    // 多次失效无收益、纯开销（空查时尤其明显）
    DRCO_DCCI_SINGLE_CACHE_LINE(&matrix->rowList[rowIdx]);
    for (uint32_t i = 0; i < wordCnt && count < maxCount; i++) {
        __gm__ uint64_t* word = DrcoLocalReadyMatrixWord(matrix, i, rowIdx);
        if (*word == 0) {
            continue;
        }
        uint64_t old = DrcoAtomicExchToU64(word, 0);
        uint32_t lo = static_cast<uint32_t>(old & 0xFFFFFFFFULL);
        uint32_t hi = static_cast<uint32_t>(old >> 32);
        if (lo != 0) {
            outTaskList[count++] = DRCO_DECODE_TASK(lo);
        }
        if (hi != 0) {
            outTaskList[count++] = DRCO_DECODE_TASK(hi);
        }
    }
    return count;
}

// ==================== LocalReadyQueue 原语 ====================

INLINE int DrcoLocalReadyQueueTryPushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t readyTask)
{
    __gm__ uint32_t* tailPtr = &queue->tail;
    uint32_t tail = DrcoAtomicLoad(tailPtr);
    if (tail >= DrcoGmLoad(&queue->size)) {
        return -1;
    }
    uint32_t tailPrev = DrcoAtomicCasToU32(&queue->tail, tail, tail + 1);
    if (tailPrev == tail) {
        DrcoAtomicExchToU32(&queue->taskList[tailPrev], DRCO_ENCODE_TASK(readyTask));
        return 0;
    }
    return 1;
}

INLINE bool DrcoLocalReadyQueuePushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t readyTask)
{
    int result = DrcoLocalReadyQueueTryPushTask(queue, readyTask);
    while (result == 1) {
        result = DrcoLocalReadyQueueTryPushTask(queue, readyTask);
    }
    return result == 0;
}

INLINE int DrcoLocalReadyQueueTryBatchPushTask(__gm__ DrcoLocalReadyQueue* queue, uint32_t* readyTaskList, uint32_t n)
{
    __gm__ uint32_t* tailPtr = &queue->tail;
    uint32_t tail = DrcoAtomicLoad(tailPtr);
    if (tail >= DrcoGmLoad(&queue->size)) {
        return -1;
    }
    uint32_t tailPrev = DrcoAtomicCasToU32(&queue->tail, tail, tail + n);
    if (tailPrev == tail) {
        for (uint32_t i = 0; i < n; i++) {
            DrcoAtomicExchToU32(&queue->taskList[tail + i], DRCO_ENCODE_TASK(readyTaskList[i]));
        }
        return 0;
    }
    return 1;
}

INLINE void DrcoLocalReadyQueuePushBatch(__gm__ DrcoLocalReadyQueue* queue, uint32_t* readyTaskList, uint32_t n)
{
    int result = DrcoLocalReadyQueueTryBatchPushTask(queue, readyTaskList, n);
    while (result == 1) {
        result = DrcoLocalReadyQueueTryBatchPushTask(queue, readyTaskList, n);
    }
}

INLINE uint32_t DrcoLocalReadyQueueGetFirstTask(__gm__ DrcoLocalReadyQueue* queue)
{
    // 空判用普通 dcci 读（非 RMW）：head/tail 仅取快照，真正认领走下方 head CAS；
    // 快照陈旧最多本轮漏取（下一轮再扫），省每轮 2 次 atomicCAS(0,0) 的 L2 原子往返
    uint32_t head = DrcoGmLoad(&queue->head);
    uint32_t tail = DrcoGmLoad(&queue->tail);
    if (head >= tail) {
        return static_cast<uint32_t>(AICORE_TASK_NO_INCOME);
    }
    uint32_t headPrev = DrcoAtomicCasToU32(&queue->head, head, head + 1);
    if (headPrev != head) {
        return static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    }
    uint32_t taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
    while (taskId == 0) {
        DrcoBusyBackOff();
        taskId = DrcoAtomicLoad(&queue->taskList[headPrev]);
    }
    return DRCO_DECODE_TASK(taskId);
}

// ==================== RootFuncList 原语 ====================

// C 端聚合本组 V 计数（1C2V ssbuf 协议，if (IS_AIC) 运行时门控）：V 增量差值入账 GM AIV
// 分片（不写 staticExecCount），代做 AIV 完成判定（Σ分片 >= 期望动态计数）并发布 FINISH_VECTOR
// 镜像。调用点 = C 自身批加 + fetch 空转每轮，help 模式下也须持续——否则本组 V 末段增量永不
// 入账、AIV flag 永不置位（挂死）。完成判定按需降频：有增量立即判（波尾零延迟）；无增量每
// DRCO_VECTOR_FINISH_POLL_INTERVAL 次兜底复判一次（末次入账读漏、他组代加置 flag 均由兜底
// 补察，最多晚 N-1 轮）
constexpr uint32_t DRCO_VECTOR_FINISH_POLL_INTERVAL = 16;

INLINE void DrcoAggregateVectorCount(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    uint32_t v0 = npu::tile_fwk::DrcoWarpExecuteCount::VCountLoad(&state->ssbuf->warpExecuteCount, 0);
    uint32_t v1 = npu::tile_fwk::DrcoWarpExecuteCount::VCountLoad(&state->ssbuf->warpExecuteCount, 1);
    uint32_t delta = (v0 - state->warpAggState.lastVCount[0]) + (v1 - state->warpAggState.lastVCount[1]);
    if (delta != 0) {
        state->warpAggState.lastVCount[0] = v0;
        state->warpAggState.lastVCount[1] = v1;
        uint32_t shardIdx = BlockDescTypedBlockIdx(state->blockDesc) % npu::tile_fwk::DRCO_COUNT_SHARD_NUM;
        DrcoAtomicAddToU32(
            &rootFuncList->devTaskCountList.shardExecCount[npu::tile_fwk::DRCO_QUEUE_AIV][shardIdx].value, delta);
    }
    if (state->warpAggState.vectorFinishPublished) {
        return;
    }
    if (delta == 0 && (state->warpAggState.vectorFinishPollTick++ & (DRCO_VECTOR_FINISH_POLL_INTERVAL - 1)) != 0) {
        return;
    }
    // flag 已置位（他组 flush/开门代加/空类型预置）→ 仅需补发镜像
    if (DrcoGmLoad(&rootFuncList->devTaskFinishFlagList.flag[npu::tile_fwk::DRCO_QUEUE_AIV].devTaskFinishFlag) == 0) {
        uint32_t total = 0;
        for (uint32_t s = 0; s < npu::tile_fwk::DRCO_COUNT_SHARD_NUM; s++) {
            total += DrcoGmLoad(&rootFuncList->devTaskCountList.shardExecCount[npu::tile_fwk::DRCO_QUEUE_AIV][s].value);
        }
        if (total < state->warpAggState.vectorCountDynExpect) {
            return;
        }
        DrcoGmStore(&rootFuncList->devTaskFinishFlagList.flag[npu::tile_fwk::DRCO_QUEUE_AIV].devTaskFinishFlag,
                    uint32_t(1));
    }
    npu::tile_fwk::DrcoWarpExecuteCount::FinishPublish(&state->ssbuf->warpExecuteCount,
                                                       npu::tile_fwk::DrcoWarpExecuteCount::FINISH_VECTOR);
    state->warpAggState.vectorFinishPublished = true;
}

// leaf task 计数并广播完成：累加本 coreType 动态分片
// 仅当 Σ分片达期望动态计数的最后一次累加方置 1 本 coreType 的 devTaskFinishFlag
// 本类型全部核经 DrcoGmLoad（含 dcci 失效）轮询同一标志
// 1C2V ssbuf 协议：V 端只写组内 ssbuf 分片（GM 原子/完成判定由同组 C 代做）；C 端追加聚合
INLINE void DrcoNotifyTaskExecutedAdd(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                      uint32_t addCount)
{
    if (addCount == 0) {
        // MQ probe-break 空批：零值原子 RMW 仍占分片计数线串行 + 双分片 DCCI 读——波尾成员
        // 核对齐时实测 4.2us（头核走 KernelEntryDrco 直消费路径绕过本调用仅 0.39us）。
        // 完成判定不受影响：末次真实 add 判定，fetch 空转侧 DrcoTryFinishByShardCount 兜底
        // （ssbuf 协议下由空转侧 DrcoAggregateVectorCount 聚合兜底）
        return;
    }
    if (IS_AIV) {
        // V 端：只写本组 ssbuf 分片；计数时机与旧路径一致（fetch 后整批认领、notify before run）
        npu::tile_fwk::DrcoWarpExecuteCount::VCountAdd(&state->ssbuf->warpExecuteCount, get_subblockid() == 1 ? 1u : 0u,
                                                       addCount);
        return;
    }
    if (IS_AIC) {
        // C 端：先聚合本组 V 增量（幂等），自身批加走随后原逻辑
        DrcoAggregateVectorCount(state, rootFuncList);
    }
    // 动态 leaf 计数分片累加：各核只加自己分片（按类型内编号散列），避免 64 路单点原子争抢
    uint32_t shardIdx = BlockDescTypedBlockIdx(state->blockDesc) % npu::tile_fwk::DRCO_COUNT_SHARD_NUM;
    DrcoAtomicAddToU32(&rootFuncList->devTaskCountList.shardExecCount[DRCO_CORE_TYPE][shardIdx].value, addCount);
    // 完成判定 = Σ分片 >= 期望动态计数（size - host 静态预累加，入口缓存）；各分片走 dcci 读。
    // 最后一次使分片和达期望值的累加方在本核 add 之后读全部分片；若因可见性读漏，fetch 空转侧
    // DrcoTryFinishByShardCount 会补判（避免漏判挂死）。空类型由 host 在 precount 阶段直接置位
    uint32_t total = 0;
    for (uint32_t s = 0; s < npu::tile_fwk::DRCO_COUNT_SHARD_NUM; s++) {
        total += DrcoGmLoad(&rootFuncList->devTaskCountList.shardExecCount[DRCO_CORE_TYPE][s].value);
    }
    if (total < state->drcoCountDynExpect) {
        return;
    }
    DrcoGmStore(&rootFuncList->devTaskFinishFlagList.flag[DRCO_CORE_TYPE].devTaskFinishFlag, uint32_t(1));
}

// 同上，目标计数器由参数指定（开门核代加场景：需替两种核型的成员分别记账，
// DRCO_CORE_TYPE 是编译期宏无法跨型，故参数化）；跨型无核本地缓存，期望动态计数由 GM
// size/静态预累加直读合成（非热路径），动态量同走分片——staticExecCount 设备侧只读，
// 代加若写回 staticExecCount 会游离在完成判定（Σ分片 >= 期望值）之外而漏判挂死
INLINE void DrcoNotifyTaskExecutedAddType(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                          uint32_t queueType, uint32_t addCount)
{
    if (addCount == 0) {
        return;
    }
    uint32_t shardIdx = BlockDescTypedBlockIdx(state->blockDesc) % npu::tile_fwk::DRCO_COUNT_SHARD_NUM;
    DrcoAtomicAddToU32(&rootFuncList->devTaskCountList.shardExecCount[queueType][shardIdx].value, addCount);
    uint32_t expect = DrcoGmLoad(&rootFuncList->devTaskCountList.count[queueType].size) -
                      DrcoGmLoad(&rootFuncList->devTaskCountList.count[queueType].staticExecCount);
    uint32_t total = 0;
    for (uint32_t s = 0; s < npu::tile_fwk::DRCO_COUNT_SHARD_NUM; s++) {
        total += DrcoGmLoad(&rootFuncList->devTaskCountList.shardExecCount[queueType][s].value);
    }
    if (total < expect) {
        return;
    }
    DrcoGmStore(&rootFuncList->devTaskFinishFlagList.flag[queueType].devTaskFinishFlag, uint32_t(1));
}

INLINE __gm__ DrcoLocalReadyQueue* DrcoRootFuncListGetLocalReadyQueue(
    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t readyQueueCoreType, uint32_t groupIdx)
{
    if (readyQueueCoreType < npu::tile_fwk::DRCO_QUEUE_MAX && groupIdx < npu::tile_fwk::NUM_LOCAL_GROUPS) {
        return rootFuncList->localReadyQueueArray[readyQueueCoreType][groupIdx];
    }
    return nullptr;
}

INLINE __gm__ DrcoLocalReadyMatrix* DrcoRootFuncListGetLocalReadyMatrix(
    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t readyMatrixCoreType)
{
    if (readyMatrixCoreType < npu::tile_fwk::DRCO_QUEUE_MAX) {
        return rootFuncList->localReadyMatrixArray[readyMatrixCoreType];
    }
    return nullptr;
}

// 全核共享 stitch 矩阵（行 = 全局 blockIdx）：AIC/AIV 任意核可 push/pop，
// 展开动作类型无关，空闲侧核帮忙消化忙侧 defer 链
INLINE __gm__ DrcoGlobalStitchNodeMatrix* DrcoRootFuncListGetStitchNodeMatrix(
    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    return rootFuncList->stitchNodeMatrix;
}

// 类型内本地编号模型（AIC: blockIdx ∈ [0, aicCoreNum)，AIV: blockIdx - aicCoreNum ∈ [0, 2*aicCoreNum)）
// 下各队列行实际使用的 local ready queue/matrix 组区间为 [groupBeg, groupEnd)：两类均自组 0 起
// （AIC/MIX 组数 ceil(aicCoreNum/N)，AIV 组数 ceil(2*aicCoreNum/N)），区间外的组既无 push 亦无 poll，
// 其消费者为 AIC 核，组区间与 AIC 行一致。
// 仅在 InitDrcoEntry 调用一次，结果缓存在 ExecuteContext::drcoGroupBeg/End，热路径直接查表
INLINE void GetDrcoActiveGroupRange(uint32_t queueType, uint32_t aicCoreNum, uint32_t& groupBeg, uint32_t& groupEnd)
{
    if (queueType == npu::tile_fwk::DRCO_QUEUE_AIV) {
        groupBeg = 0;
        groupEnd = (aicCoreNum * 2 + npu::tile_fwk::LOCAL_GROUP_SIZE - 1) / npu::tile_fwk::LOCAL_GROUP_SIZE;
    } else if (queueType == npu::tile_fwk::DRCO_QUEUE_AIC || queueType == npu::tile_fwk::DRCO_QUEUE_MIX) {
        groupBeg = 0;
        groupEnd = (aicCoreNum + npu::tile_fwk::LOCAL_GROUP_SIZE - 1) / npu::tile_fwk::LOCAL_GROUP_SIZE;
    } else {
        groupBeg = 0;
        groupEnd = npu::tile_fwk::NUM_LOCAL_GROUPS;
    }
    if (groupEnd > npu::tile_fwk::NUM_LOCAL_GROUPS) {
        groupEnd = npu::tile_fwk::NUM_LOCAL_GROUPS;
    }
}

INLINE __gm__ DrcoLocalReadyQueue* DrcoDynFuncDataListGetIdleQueue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                                   uint32_t succCoreType, uint32_t groupCount,
                                                                   uint32_t startOff)
{
    for (uint32_t i = 0; i < groupCount; i++) {
        uint32_t groupIdx = (startOff + i) % groupCount;
        __gm__ DrcoLocalReadyQueue* queue = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, succCoreType, groupIdx);
        if (queue == nullptr) {
            continue;
        }
        uint32_t head = DrcoAtomicLoad(&queue->head);
        uint32_t tail = DrcoAtomicLoad(&queue->tail);
        if (head >= tail) {
            return queue;
        }
    }
    return nullptr;
}

INLINE uint32_t DrcoDynFuncDataListPushBatchLocalMatrix(DrcoEntryState* state,
                                                        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                        uint32_t* succTaskIdList, uint32_t succTaskIdListSize,
                                                        uint32_t succCoreType, uint32_t typedBlockIdx)
{
    return DrcoLocalReadyMatrixPushBatch(state, rootFuncList, typedBlockIdx, succTaskIdList, succTaskIdListSize,
                                         succCoreType);
}

INLINE void DrcoDynFuncDataListPushBatchLocalQueue(DrcoEntryState* state,
                                                   __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                   uint32_t* succTaskIdList, uint32_t succTaskIdListSize,
                                                   uint32_t succCoreType, uint32_t groupCount, uint32_t typedBlockIdx)
{
    // 剩余任务溢出到本类型 local ready queue：
    // 份额递减分配 ceil(size/2)→当前队列、ceil(size/4)→+1 队列、ceil(size/8)→+2 …，
    // 当前队列整批放不下则环形换下一个 group；递减份额取尽后，最后不足一批的剩余全部放进当前队列
    uint32_t groupStart = typedBlockIdx / npu::tile_fwk::LOCAL_GROUP_SIZE;
    uint32_t groupIdx = groupStart;
    uint32_t batchTaskIdx = 0;
    uint32_t rem = succTaskIdListSize;
    uint32_t shift = 1;
    while (rem > 0) {
        uint32_t batch = (succTaskIdListSize + (1u << shift) - 1) >> shift;
        if (batch > rem) {
            batch = rem;
        }
        bool batchPushed = false;
        for (uint32_t g = 0; g < groupCount; g++) {
            uint32_t queueIdx = (groupIdx + g) % groupCount;
            __gm__ DrcoLocalReadyQueue* localQueue = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, succCoreType,
                                                                                        queueIdx);
            // 原子预留 tail..tail+batch-1；容量不足或被其他核抢满则换下一个队列
            while (true) {
                uint32_t tail = DrcoAtomicLoad(&localQueue->tail);
                if (tail + batch > DrcoGmLoad(&localQueue->size)) {
                    break;
                }
                uint32_t tailPrev = DrcoAtomicCasToU32(&localQueue->tail, tail, tail + batch);
                if (tailPrev == tail) {
                    for (uint32_t i = 0; i < batch; i++) {
                        uint32_t succTaskId = succTaskIdList[batchTaskIdx + i];
                        DrcoAtomicExchToU32(&localQueue->taskList[tail + i], DRCO_ENCODE_TASK(succTaskId));
                        TraceEvent(state, succTaskId, EVENT_PUSH_LOCAL(queueIdx));
                    }
                    batchTaskIdx += batch;
                    rem -= batch;
                    groupIdx = (queueIdx + 1) % groupCount;
                    batchPushed = true;
                    break;
                }
            }
            if (batchPushed) {
                break;
            }
        }
        if (!batchPushed) {
            break;
        }
        shift++;
    }
}

INLINE void DrcoDynFuncDataListPushBatch(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                         uint32_t* succTaskIdList, uint32_t succTaskIdListSize, uint32_t succCoreType)
{
    BlockDesc blockDesc = state->blockDesc;
    uint32_t typedBlockIdx = BlockDescTypedBlockIdx(blockDesc);

    uint32_t pushed = DrcoDynFuncDataListPushBatchLocalMatrix(state, rootFuncList, succTaskIdList, succTaskIdListSize,
                                                              succCoreType, typedBlockIdx);
    succTaskIdListSize -= pushed;

    if (succTaskIdListSize > 0) {
        // 溢出到本类型 local ready queue（仍按组分池）
        uint32_t groupCount = state->ctx.drcoGroupEnd[succCoreType] - state->ctx.drcoGroupBeg[succCoreType];
        DrcoDynFuncDataListPushBatchLocalQueue(state, rootFuncList, succTaskIdList + pushed, succTaskIdListSize,
                                               succCoreType, groupCount, typedBlockIdx);
    }
}

INLINE void DrcoFlushBatchTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                uint32_t succTaskIdListSizeCoreList[])
{
    for (uint32_t coreType = 0; coreType < npu::tile_fwk::DRCO_QUEUE_MAX; coreType++) {
        if (succTaskIdListSizeCoreList[coreType] > 0) {
            // 直通 selfReadyQueue：同类型（succ core type == 本核类型）后继优先直通本核 selfReadyQueue，
            // 本核 fetch 下一轮开头直接全部执行，省 push + 矩阵/队列探测往返；
            // selfReadyQueue 放不下的溢出部分再走 localMatrix/localQueue
            uint32_t directCnt = 0;
            if (coreType == DRCO_CORE_TYPE && state->selfReadyQueueSize < SELF_QUEUE_SIZE) {
                uint32_t freeCnt = SELF_QUEUE_SIZE - state->selfReadyQueueSize;
                directCnt = succTaskIdListSizeCoreList[coreType] < freeCnt ? succTaskIdListSizeCoreList[coreType] :
                                                                             freeCnt;
                for (uint32_t i = 0; i < directCnt; i++) {
                    uint32_t succTaskId = succTaskIdListCoreList[coreType][i];
                    state->selfReadyQueue[state->selfReadyQueueSize + i] = succTaskId;
                    TraceEvent(state, succTaskId, EVENT_PUSH_SELF_READY_QUEUE());
                }
                state->selfReadyQueueSize += directCnt;
            }
            if (directCnt < succTaskIdListSizeCoreList[coreType]) {
                DrcoDynFuncDataListPushBatch(state, rootFuncList, succTaskIdListCoreList[coreType] + directCnt,
                                             succTaskIdListSizeCoreList[coreType] - directCnt, coreType);
            }
            succTaskIdListSizeCoreList[coreType] = 0;
        }
    }
}

// 尝试把单个兜底 hub 任务 CAS 写入全核共享矩阵：从 (push 核行 + start) 起逐行抢占空槽，
// 成功返回 true 并推进 start（后续改投行分散、减少竞争）；起点跳过 push 核自己的行——
// producer 正忙于解依赖无暇 pop 本行，落其他行空闲核立即可并行消费
// slot 存 DRCO_ENCODE_TASK 的 taskId（0 = 空闲），与 local matrix 同款编码
INLINE static bool DrcoHubTaskMatrixTryPush(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                            uint32_t taskId, uint32_t& start)
{
    __gm__ npu::tile_fwk::DrcoGlobalHubTaskMatrix* matrix = rootFuncList->hubTaskMatrix;
    if (matrix == nullptr) {
        // host 恒分配（InitDrcoRootFuncList），null 即内部状态错误，显性失败优于静默丢任务
        Trap();
        return false;
    }
    uint32_t rowCnt = state->ctx.aicCoreNum * DRCO_ALL_CORES_PER_AIC;
    uint32_t coreTypeIdx = BlockDescBlockIdx(state->blockDesc);
    uint32_t colIdx = coreTypeIdx % npu::tile_fwk::DrcoGlobalHubTaskMatrix::COL_SIZE;
    for (uint32_t i = 0; i < rowCnt; i++) {
        uint32_t rowIdx = (coreTypeIdx + start + i) % rowCnt;
        __gm__ uint32_t* slot = &matrix->hubTaskList[rowIdx].slot[colIdx];
        uint32_t prev = DrcoAtomicCasToU32(slot, 0, DRCO_ENCODE_TASK(taskId));
        if (prev == 0) {
            start = start + i + 1;
            return true;
        }
    }
    return false;
}

// 矩阵整满时自旋重试：全核在 fetch 循环无条件扫 hub 矩阵，矩阵内任务被消费必有进展；
// 超时 Trap 兜底，杜绝静默丢任务导致的挂死（同 ResolveHubMixTask 的 C2V push 模式）
INLINE void DrcoHubTaskMatrixPush(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                  uint32_t taskId)
{
    uint32_t start = 1;
    uint64_t pushStart = get_sys_cnt();
    while (!DrcoHubTaskMatrixTryPush(state, rootFuncList, taskId, start)) {
        if (get_sys_cnt() - pushStart > AICORE_LEAF_TASK_RUN_TIMEOUT) {
            Trap();
        }
        DrcoBusyBackOff();
    }
}

INLINE void DrcoResolveDependOnce(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                  uint32_t succTaskId, uint32_t hubStack[], int32_t& hubStackTop,
                                  uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                  uint32_t succTaskIdListSizeCoreList[])
{
    TraceEvent(state, succTaskId, EVENT_RESOLVE());
    // taskId 解码出的 coreType（AIV/AIC/HUB_MIX）与 DRCO_QUEUE_* 取值一一对应，直接作队列下标
    uint32_t succQueueType = npu::tile_fwk::DrcoTaskCoreTypeOf(succTaskId);
    if (succQueueType < npu::tile_fwk::DRCO_QUEUE_MAX) {
        succTaskIdListCoreList[succQueueType][succTaskIdListSizeCoreList[succQueueType]++] = succTaskId;
        if (succTaskIdListSizeCoreList[succQueueType] >= BATCH_PUSH_BUF_SIZE) {
            DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
        }
    } else {
        FedProfLeafTaskProfRecordMix(&state->ctx, succTaskId, npu::tile_fwk::FuncID(succTaskId));
        // hubStack == nullptr（消费核 pop 路径）或栈满：改投全核共享 hub 矩阵，被任意核
        // fetch 时 pop 就地解依赖（不执行不计数）；own-type 队列因此只含可执行 leaf
        if (hubStack != nullptr && hubStackTop + 1 < HUB_STACK_SIZE) {
            hubStack[++hubStackTop] = succTaskId;
        } else {
            DrcoHubTaskMatrixPush(state, rootFuncList, succTaskId);
        }
    }
}

// stitch 节点 defer 消费侧专用：stitch 后继不出现 hub，无需 hubStack 分支，
// 按核类型 batch push 即可（类型越界改投全核共享 hub 矩阵兜底，防数组越界）
INLINE void DrcoResolveDependOnceCore(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                      uint32_t succTaskId, uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                      uint32_t succTaskIdListSizeCoreList[])
{
    TraceEvent(state, succTaskId, EVENT_RESOLVE());
    // 同 DrcoResolveDependOnce：解码出的 coreType 即队列下标（HUB_MIX 对应 MIX 行）；
    // stitch 后继按约定不出现 hub，万一出现 HUB_MIX 也走 mixhub 派发路径而非被当 leaf 执行
    uint32_t succQueueType = npu::tile_fwk::DrcoTaskCoreTypeOf(succTaskId);
    if (succQueueType < npu::tile_fwk::DRCO_QUEUE_MAX) {
        succTaskIdListCoreList[succQueueType][succTaskIdListSizeCoreList[succQueueType]++] = succTaskId;
        if (succTaskIdListSizeCoreList[succQueueType] >= BATCH_PUSH_BUF_SIZE) {
            DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
        }
    } else {
        // 类型越界兜底：同 hub 溢出路径改投全核共享 hub 矩阵（pop 后就地解依赖，不执行不计数），
        // own-type 队列因此只含可执行 leaf，ExecDrcoReadyQueueTasks 无需 FIN/hub 预判
        DrcoHubTaskMatrixPush(state, rootFuncList, succTaskId);
    }
}

// 解码 stitch 节点 nodeNext 低 6 位：本节点之后的节点数 R（子链大小 S = R + 1，R ≤ 63）。
// 链头写 min(N-1,63)；每个子链头写其子链后续节点数（EncodeStitchNodes 写入）。
// 任务数始终从 nodeSize 字段读取（不参与编码）。
INLINE static uint32_t StitchNodeRemainingCount(__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node)
{
    return static_cast<uint32_t>(reinterpret_cast<uint64_t>(node->nodeNext) &
                                 npu::tile_fwk::DUPPED_STITCH_NODE_REMAIN_COUNT_MASK);
}

// device 侧遍历解码：nodeNext 低 6 位含编码，清低 6 位后返回真实后继地址
INLINE static __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* StitchNodeDecodedNext(
    __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node)
{
    return reinterpret_cast<__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode*>(
        reinterpret_cast<uint64_t>(node->nodeNext) & npu::tile_fwk::DUPPED_STITCH_NODE_ADDR_MASK);
}

// 就地遍历单个 stitch 节点解依赖：DrcoResolveDepend 首节点路径与矩阵 push 失败兜底共用，
// 走完整 DrcoResolveDependOnce 路由（含 hub 处理）
INLINE void DrcoResolveStitchNodeTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                       __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node,
                                       uint32_t curTaskId, uint32_t hubStack[], int32_t& hubStackTop,
                                       uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                       uint32_t succTaskIdListSizeCoreList[])
{
    TraceEvent(state, curTaskId, EVENT_SUCC_DYNAMIC(node->nodeSize));
    __gm__ void* controlFlowCacheBase = state->controlFlowCacheBase;
    __gm__ void* metadataBase = state->metadataBase;
    for (uint32_t i = 0; i < node->nodeSize; i++) {
        uint32_t succTaskId = node->nodeTaskList[i]; // already coreType-encoded at build time
        uint32_t succFuncId = npu::tile_fwk::FuncID(succTaskId);
        uint32_t succOpIdx = npu::tile_fwk::TaskID(succTaskId);
        __gm__ npu::tile_fwk::DrcoRootFuncData* succRootFuncData = DrcoGetCurrentRootFuncData(state, succFuncId);
        uint64_t funcDynamicDataOffset = succRootFuncData->funcDynamicDataOffset;
        __gm__ int32_t* succPredCount = DrcoGetCurrentRootFuncPredCount(controlFlowCacheBase, metadataBase,
                                                                        static_cast<uint32_t>(funcDynamicDataOffset));
        int32_t old = DrcoAtomicResolveDependOnce<int32_t, 1, -1>(&succPredCount[succOpIdx]);
        if (old == 1) {
            DrcoResolveDependOnce(state, rootFuncList, succTaskId, hubStack, hubStackTop, succTaskIdListCoreList,
                                  succTaskIdListSizeCoreList);
        }
    }
}

// Fire a claimed successor: decode the drco-encoded entry into a taskId and route it
// through the full DrcoResolveDependOnce path (incl. hub handling).
INLINE void DrcoFireEncodedSucc(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                uint32_t funcIdx, uint32_t succEncoded, uint32_t hubStack[], int32_t& hubStackTop,
                                uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                uint32_t succTaskIdListSizeCoreList[])
{
    uint32_t succOpIdx = succEncoded & TASKID_TASK_MASK;
    uint32_t succCoreType = (succEncoded >> npu::tile_fwk::TASKID_DRCO_CT_SHIFT) & npu::tile_fwk::TASKID_DRCO_CT_MASK;
    uint32_t succTaskId = npu::tile_fwk::MakeDrcoTaskId(funcIdx, succOpIdx, succCoreType);
    DrcoResolveDependOnce(state, rootFuncList, succTaskId, hubStack, hubStackTop, succTaskIdListCoreList,
                          succTaskIdListSizeCoreList);
}

INLINE bool DrcoResolveDependDummyEnding(uint32_t succEncoded)
{
    return succEncoded & npu::tile_fwk::DRCO_SUCC_DUMMY_ENDING_BIT;
}

INLINE bool DrcoResolveDependMemoryReuseStitched(uint64_t funcInfoAndStaticDataOffset)
{
    return funcInfoAndStaticDataOffset & npu::tile_fwk::DRCO_ROOT_FUNC_MEMORY_REUSE_STITCHED_BIT;
}

// 解一段 per-func 相对编码后继条目（平铺区 / 节点 succList 同构）：配对 (2k, 2k+1) 共享
// 一次 u64 原子减（恰好一次解依赖保证不借位，old==1 的半边唯一过零触发 fire）；单条走
// i32，dummy ending 且本 func 未内存复用缝合时跳过递减（同上游平铺语义）。predCount 由
// funcId 一次索引整段复用
INLINE static void DrcoResolveEncodedSuccList(DrcoEntryState* state,
                                              __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t funcIdx,
                                              __gm__ uint32_t* succEncodedList, uint32_t begin, uint32_t end,
                                              uint32_t hubStack[], int32_t& hubStackTop,
                                              uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                              uint32_t succTaskIdListSizeCoreList[])
{
    __gm__ npu::tile_fwk::DrcoRootFuncData* rootFuncData = DrcoGetCurrentRootFuncData(state, funcIdx);
    __gm__ int32_t* predCount = DrcoGetCurrentRootFuncPredCount(
        state->controlFlowCacheBase, state->metadataBase, static_cast<uint32_t>(rootFuncData->funcDynamicDataOffset));
    bool memReuseStitched = DrcoResolveDependMemoryReuseStitched(rootFuncData->funcInfoAndStaticDataOffset);
    for (uint32_t i = begin; i < end;) {
        uint32_t succEncoded = succEncodedList[i];
        uint32_t succOpIdx = succEncoded & TASKID_TASK_MASK;
        if ((succEncoded & npu::tile_fwk::DRCO_SUCC_PAIR_BIT) != 0) {
            uint32_t nextEncoded = succEncodedList[i + 1];
            __gm__ uint64_t* slot = reinterpret_cast<__gm__ uint64_t*>(&predCount[succOpIdx]);
            uint64_t old = DrcoAtomicResolveDependOnce<uint64_t, npu::tile_fwk::DRCO_SUCC_PAIR_BOTH_ONE,
                                                       npu::tile_fwk::DRCO_SUCC_PAIR_DEC>(slot);
            if (static_cast<uint32_t>(old) == 1u) {
                DrcoFireEncodedSucc(state, rootFuncList, funcIdx, succEncoded, hubStack, hubStackTop,
                                    succTaskIdListCoreList, succTaskIdListSizeCoreList);
            }
            if (static_cast<uint32_t>(old >> 32) == 1u) {
                DrcoFireEncodedSucc(state, rootFuncList, funcIdx, nextEncoded, hubStack, hubStackTop,
                                    succTaskIdListCoreList, succTaskIdListSizeCoreList);
            }
            i += 2;
        } else {
            if (DrcoResolveDependDummyEnding(succEncoded) && !memReuseStitched) {
                // Successor is dummy ending, and current is never memory reuse stitched to following root
                // functions.
            } else {
                int32_t old = DrcoAtomicResolveDependOnce<int32_t, 1, -1>(&predCount[succOpIdx]);
                if (old == 1) {
                    DrcoFireEncodedSucc(state, rootFuncList, funcIdx, succEncoded, hubStack, hubStackTop,
                                        succTaskIdListCoreList, succTaskIdListSizeCoreList);
                }
            }
            i += 1;
        }
    }
}

// ==================== 静态后继节点 push（独立于 stitch 节点 push） ====================

// 节点 coreType → stitch matrix 行域（游标域内绕接，绝不跨类型）：AIV 节点进 AIV 行域
// [nrValidAic, 3*nrValidAic)，AIC/HUB_MIX/HUB 节点进 AIC 行域 [0, nrValidAic)
INLINE static void DrcoStaticSuccNodeRowRange(uint32_t nodeCoreType, uint32_t nrValidAic, uint32_t& rangeBegin,
                                              uint32_t& rangeSize)
{
    if (nodeCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIV)) {
        rangeBegin = nrValidAic;
        rangeSize = (DRCO_ALL_CORES_PER_AIC - 1) * nrValidAic;
    } else {
        rangeBegin = 0;
        rangeSize = nrValidAic;
    }
}

// 节点 coreType → 行域对应的 readymatrix 队列下标（锚点游标选取用）
INLINE static uint32_t DrcoStaticSuccNodeQueueType(uint32_t nodeCoreType)
{
    return nodeCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIV) ? npu::tile_fwk::DRCO_QUEUE_AIV :
                                                                                 npu::tile_fwk::DRCO_QUEUE_AIC;
}

// 就地解单个静态后继节点：生产核（HUB 节点本地解 / push 失败兜底）传真实 hubStack；
// 消费核传 nullptr——pop 到的节点不含 HUB（HUB 节点从不入 matrix），不会被触达
// （nullptr 守卫兜底）
INLINE void DrcoResolveStaticSuccNodeTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                           uint32_t funcIdx,
                                           __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* node,
                                           uint32_t hubStack[], int32_t& hubStackTop,
                                           uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                           uint32_t succTaskIdListSizeCoreList[])
{
    DrcoResolveEncodedSuccList(state, rootFuncList, funcIdx, node->succList, 0, node->succSize, hubStack, hubStackTop,
                               succTaskIdListCoreList, succTaskIdListSizeCoreList);
}
// 尝试把 slot 写入矩阵：从本核行 start 偏移起逐行 CAS 抢空槽（0 = 空闲），成功推进
// start（同链节点行分散）。slot（u32）编码见 aikernel_device_task.h
INLINE static bool DrcoStitchNodeMatrixTryPushSlot(DrcoEntryState* state, __gm__ DrcoGlobalStitchNodeMatrix* matrix,
                                                   uint32_t slotValue, uint32_t& start)
{
    BlockDesc blockDesc = state->blockDesc;
    // 行 = 全局 blockIdx（AIC [0,nrValidAic) + AIV [nrValidAic,3*nrValidAic)）：
    // 全核共享池，节点落在任意空行即被该行主人 pop 展开（类型无关）
    uint32_t rowCnt = state->ctx.aicCoreNum * DRCO_ALL_CORES_PER_AIC;
    uint32_t coreTypeIdx = BlockDescBlockIdx(blockDesc);
    uint32_t colIdx = coreTypeIdx % npu::tile_fwk::DrcoGlobalStitchNodeMatrix::COL_SIZE;
    for (uint32_t i = 0; i < rowCnt; i++) {
        // 起点 +1 跳过 push 核自己的行：producer push 完即投入执行/继续解依赖，无暇
        // pop 本行；子链头落在空闲核行可立即被接力展开（与 hub matrix 同款策略——
        // 落本核行时后续批后继的解依赖会被当前批执行整段阻塞）
        uint32_t rowIdx = (coreTypeIdx + 1 + start + i) % rowCnt;
        __gm__ uint32_t* slot = &matrix->stitchNodeList[rowIdx].slot[colIdx];
        uint32_t prev = DrcoAtomicCasToU32(slot, 0, slotValue);
        if (prev == 0) {
            start = start + i + 1;
            return true;
        }
    }
    return false;
}

// stitch 链节点 push：slot = 节点相对 stitchNodeBase 的 u32 偏移（bit31 = 0 → 非静态）
INLINE static bool DrcoStitchNodeMatrixTryPush(DrcoEntryState* state,
                                               __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                               __gm__ DrcoGlobalStitchNodeMatrix* matrix,
                                               __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node,
                                               uint32_t& start)
{
    uint32_t nodeOffset = static_cast<uint32_t>(reinterpret_cast<uint64_t>(node) - rootFuncList->stitchNodeBase);
    return DrcoStitchNodeMatrixTryPushSlot(state, matrix, nodeOffset, start);
}

// 静态后继节点专用 push（与 stitch 节点 push 完全独立，stitch 侧行为零改动）：行域按节点
// coreType 收窄，start 为域内游标（CAS 抢行推进，域满返回 false 由调用方就地解兜底）；
// slot 携带 per-func 节点下标（消费核按 funcId 取节点区基址 + 下标还原）
INLINE static bool DrcoStaticSuccNodeMatrixTryPush(DrcoEntryState* state, __gm__ DrcoGlobalStitchNodeMatrix* matrix,
                                                   __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* node,
                                                   uint32_t funcId, uint32_t nodeIndex, uint32_t& start)
{
    uint32_t slotValue = npu::tile_fwk::EncodeStaticSuccSlot(funcId, nodeIndex);
    uint32_t rangeBegin;
    uint32_t rangeSize;
    DrcoStaticSuccNodeRowRange(node->coreType, state->ctx.aicCoreNum, rangeBegin, rangeSize);
    uint32_t colIdx = BlockDescBlockIdx(state->blockDesc) % npu::tile_fwk::DrcoGlobalStitchNodeMatrix::COL_SIZE;
    for (uint32_t i = 0; i < rangeSize; i++) {
        uint32_t rowIdx = rangeBegin + (start + i) % rangeSize;
        __gm__ uint32_t* slot = &matrix->stitchNodeList[rowIdx].slot[colIdx];
        if (DrcoAtomicCasToU32(slot, 0, slotValue) == 0) {
            start = (start + i + 1) % rangeSize; // 域内绕接（不跨类型）
            return true;
        }
    }
    return false;
}

// 认领本表各队列类型的散射行起点：单遍统计各类型非 HUB 节点数（1 节点 1 行），逐类型经全局游标
// readyMatrixPushCol 原子占位后域内绕接——与矩阵 push 共用同一串行序列，直通项固定落在这些行上，
// 两通道行账统一、互不踩行。起点/行域写回 succStarts[]/succRanges[]；供 DrcoResolveStaticSuccNodes
// 在函数开头一次性取用，避免循环内反复扫描 nodeSize
INLINE static void DrcoStaticSuccNodeClaimRows(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                               __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* nodes,
                                               uint16_t staticSuccNodeSize, uint32_t aicCoreNum,
                                               uint32_t succStarts[npu::tile_fwk::DRCO_QUEUE_MAX],
                                               uint32_t succRanges[npu::tile_fwk::DRCO_QUEUE_MAX])
{
    uint32_t nodeCnt[npu::tile_fwk::DRCO_QUEUE_MAX] = {0};
    for (uint32_t m = 0; m < staticSuccNodeSize; m++) {
        if (nodes[m].coreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB)) {
            continue;
        }
        nodeCnt[DrcoStaticSuccNodeQueueType(nodes[m].coreType)]++;
    }
    for (uint32_t q = 0; q < npu::tile_fwk::DRCO_QUEUE_MAX; q++) {
        if (nodeCnt[q] == 0) {
            continue;
        }
        uint32_t nodeCoreType = (q == npu::tile_fwk::DRCO_QUEUE_AIV) ?
                                    static_cast<uint32_t>(npu::tile_fwk::CoreType::AIV) :
                                    static_cast<uint32_t>(npu::tile_fwk::CoreType::AIC);
        uint32_t rangeBegin = 0;
        DrcoStaticSuccNodeRowRange(nodeCoreType, aicCoreNum, rangeBegin, succRanges[q]);
        succStarts[q] = DrcoAtomicAddToU32(&rootFuncList->readyMatrixPushCol[q], nodeCnt[q]) % succRanges[q];
    }
}

// 就地解单个 stitch 节点依赖：走 DrcoResolveDependOnceCore 路由（无 hub 级联，batch push），
// 任务数从编码后的 nodeSize 解码；量化指针经 metadataBase 还原；消费核 pop 与生产核 push 失败兜底共用
INLINE void DrcoStitchNodeTasksResolveCore(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                           __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* node,
                                           uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                           uint32_t succTaskIdListSizeCoreList[])
{
    __gm__ void* controlFlowCacheBase = state->controlFlowCacheBase;
    __gm__ void* metadataBase = state->metadataBase;
    for (uint32_t i = 0; i < node->nodeSize; i++) {
        uint32_t succTaskId = node->nodeTaskList[i]; // already coreType-encoded at build time
        uint32_t succFuncId = npu::tile_fwk::FuncID(succTaskId);
        uint32_t succOpIdx = npu::tile_fwk::TaskID(succTaskId);
        __gm__ npu::tile_fwk::DrcoRootFuncData* succRootFuncData = DrcoGetCurrentRootFuncData(state, succFuncId);
        uint64_t funcDynamicDataOffset = succRootFuncData->funcDynamicDataOffset;
        __gm__ int32_t* succPredCount = DrcoGetCurrentRootFuncPredCount(controlFlowCacheBase, metadataBase,
                                                                        static_cast<uint32_t>(funcDynamicDataOffset));
        int32_t old = DrcoAtomicResolveDependOnce<int32_t, 1, -1>(&succPredCount[succOpIdx]);
        if (old == 1) {
            DrcoResolveDependOnceCore(state, rootFuncList, succTaskId, succTaskIdListCoreList,
                                      succTaskIdListSizeCoreList);
        }
    }
}

// 尝试把子链头 subChainHead 推入矩阵，失败则就地解该子链整段兜底。
// 子链边界由 subChainHead 编码的 R 确定（S = R + 1 个节点，走满 S 个即停，不越界到链尾）。
INLINE void DrcoStitchNodeTryPushChain(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                       __gm__ DrcoGlobalStitchNodeMatrix* matrix,
                                       __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* subChainHead,
                                       uint32_t& rowCursor, uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                       uint32_t succTaskIdListSizeCoreList[])
{
    if (subChainHead == nullptr) {
        return;
    }
    if (DrcoStitchNodeMatrixTryPush(state, rootFuncList, matrix, subChainHead, rowCursor)) {
        return;
    }
    uint32_t count = StitchNodeRemainingCount(subChainHead) + 1;
    for (__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* cur = subChainHead; cur != nullptr && count > 0;
         cur = StitchNodeDecodedNext(cur), count--) {
        DrcoStitchNodeTasksResolveCore(state, rootFuncList, cur, succTaskIdListCoreList, succTaskIdListSizeCoreList);
    }
}

// 按 2 的幂序列（1,2,4,8,16,32 / 每 64 一轮）把 subChainHead 起的子链头 push 入矩阵，由消费核 pop 接力。
// limitNodeCount 为本链头之后的后续节点数上限：整链展开传 0xFFFFFFFF（沿链走到 null 自然终止），
// 子链展开传 R(node)。nodeStride 翻倍推进；推进到最后一个子链头（下一 nodeStride 越过限界）即停，
// 不再沿链走到子链尾——最大子链（长度约为当前子链的一半）内部由 pop 它的消费核继续展开，避免重复遍历。
INLINE void DrcoStitchNodePushSubChainHeads(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                            __gm__ DrcoGlobalStitchNodeMatrix* matrix,
                                            __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* subChainHead,
                                            uint32_t limitNodeCount, uint32_t& rowCursor,
                                            uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
                                            uint32_t succTaskIdListSizeCoreList[])
{
    uint32_t nodeStride = 1;
    while (subChainHead != nullptr && nodeStride <= limitNodeCount) {
        DrcoStitchNodeTryPushChain(state, rootFuncList, matrix, subChainHead, rowCursor, succTaskIdListCoreList,
                                   succTaskIdListSizeCoreList);
        uint32_t nextStride = (nodeStride < 32) ? (nodeStride << 1) : 64;
        if (nextStride > limitNodeCount) {
            break;
        }
        for (uint32_t i = 0; i < nodeStride && subChainHead != nullptr; i++) {
            subChainHead = StitchNodeDecodedNext(subChainHead);
        }
        nodeStride = nextStride;
    }
}

// 生产核处理 chunked op 的整个静态后继节点表：可执行节点全部散射由消费核并行解（失败
// 就地兜底）；HUB 节点（encode 殿后）不散射——本地解，fired join 压 hubStack 由外层
// 循环展开（省 2 跳矩阵异步）。节点游标按类型分域：首节点锚定 readymatrix 游标，后续
// 按 succSize 偏移避开上一节点的任务带。succNodeIndex = 本表在 func 节点区的起始下标
// （slot 携带 per-func 节点下标，消费核按 funcId 取基址 + 下标还原）
INLINE static void DrcoResolveStaticSuccNodes(
    DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t funcIdx, uint32_t curTaskId,
    __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* nodes, uint16_t succNodeIndex, uint16_t staticSuccNodeSize,
    uint32_t hubStack[], int32_t& hubStackTop, uint32_t succTaskIdListCoreList[][BATCH_PUSH_BUF_SIZE],
    uint32_t succTaskIdListSizeCoreList[])
{
    __gm__ DrcoGlobalStitchNodeMatrix* matrix = DrcoRootFuncListGetStitchNodeMatrix(rootFuncList);
    uint32_t succNodeStarts[npu::tile_fwk::DRCO_QUEUE_MAX] = {0};
    uint32_t succNodeRanges[npu::tile_fwk::DRCO_QUEUE_MAX] = {0};
    // 函数开头一次性算好各类型散射行起点（1 节点 1 行），避免循环内反复扫描 nodeSize
    DrcoStaticSuccNodeClaimRows(rootFuncList, nodes, staticSuccNodeSize, state->ctx.aicCoreNum, succNodeStarts,
                                succNodeRanges);
    for (uint32_t n = 0; n < staticSuccNodeSize; n++) {
        if (nodes[n].coreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::HUB)) {
            DrcoResolveStaticSuccNodeTasks(state, rootFuncList, funcIdx, &nodes[n], hubStack, hubStackTop,
                                           succTaskIdListCoreList, succTaskIdListSizeCoreList);
            continue;
        }
        uint32_t nodeQueue = DrcoStaticSuccNodeQueueType(nodes[n].coreType);
        uint32_t start = succNodeStarts[nodeQueue];
        if (!DrcoStaticSuccNodeMatrixTryPush(state, matrix, &nodes[n], funcIdx, succNodeIndex + n, start)) {
            DrcoResolveStaticSuccNodeTasks(state, rootFuncList, funcIdx, &nodes[n], hubStack, hubStackTop,
                                           succTaskIdListCoreList, succTaskIdListSizeCoreList);
        }
        // 下一节点固定取认领序列的下一位（与认领行严格对齐，不用扫描推进值）
        succNodeStarts[nodeQueue] = (succNodeStarts[nodeQueue] + 1) % succNodeRanges[nodeQueue];
    }
    TraceEvent(state, curTaskId, EVENT_SUCC_STATIC(staticSuccNodeSize));
}

// 生产核整链幂分解：先按 2 的幂序列（1,2,4,8,16,32 / 每 64 一轮）把子链头 push 入矩阵
// （由消费核 pop 接力），push 完成后再解链头（chainHead）依赖（先 push 后解依赖）；
// push 失败则就地解对应子链整段兜底（矩阵写不进则保证前向推进）。
static __aicore__ void DrcoResolveDepend(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                         uint32_t* taskIdList, uint32_t taskCount = 1)
{
    uint32_t succTaskIdListCoreList[npu::tile_fwk::DRCO_QUEUE_MAX][BATCH_PUSH_BUF_SIZE];
    uint32_t succTaskIdListSizeCoreList[npu::tile_fwk::DRCO_QUEUE_MAX] = {0};

    uint32_t hubStack[HUB_STACK_SIZE];
    int32_t hubStackTop = -1;
    for (uint32_t i = 0; i < taskCount; i++) {
        hubStack[++hubStackTop] = taskIdList[i];
    }
    __gm__ void* controlFlowCacheBase = state->controlFlowCacheBase;
    __gm__ void* metadataBase = state->metadataBase;
    __gm__ void* programBase = state->programBase;
    while (hubStackTop >= 0) {
        uint32_t curTaskId = hubStack[hubStackTop--];
        if (DrcoMiddleQueueRegionHeadTryOpen(state, rootFuncList, curTaskId)) {
            continue;
        }
        uint32_t funcIdx = npu::tile_fwk::FuncID(curTaskId);
        uint32_t operIdx = npu::tile_fwk::TaskID(curTaskId);

        __gm__ npu::tile_fwk::DrcoRootFuncData* rootFuncData = DrcoGetCurrentRootFuncData(state, funcIdx);
        uint64_t funcInfoAndStaticDataOffset = rootFuncData->funcInfoAndStaticDataOffset;
        uint64_t funcDynamicDataOffset = rootFuncData->funcDynamicDataOffset;

        __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfoList = DrcoGetCurrentRootFuncSuccInfoList(
            programBase, static_cast<uint32_t>(funcInfoAndStaticDataOffset & 0x3FFFFFFFULL));
        __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode** succStitchList = DrcoGetCurrentRootFuncSuccStitchList(
            controlFlowCacheBase, metadataBase, static_cast<uint32_t>(funcDynamicDataOffset >> 32));

        volatile __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* succInfo = &succInfoList[operIdx];
        uint64_t succInfoBits = succInfo->staticIndexSizeAndStitchIndex;

        uint16_t succNodeIndex = static_cast<uint16_t>(succInfoBits);
        uint16_t succNodeSize = static_cast<uint16_t>(succInfoBits >> 16);
        __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* nodes = DrcoGetCurrentRootFuncStaticSuccNodeList(
                                                                           programBase, funcInfoAndStaticDataOffset) +
                                                                       succNodeIndex;
        if ((succInfoBits & npu::tile_fwk::DRCO_SUCC_PARALLEL_BIT) == 0) {
            // 就地串行解（小扇出/入口/HUB_MIX 生产者）：逐节点就地解，不散射不 seed 游标
            // （与平铺时代行为严格一致；trace 口径保持后继条目数）
            uint32_t succCnt = 0;
            for (uint32_t n = 0; n < succNodeSize; n++) {
                succCnt += nodes[n].succSize;
                DrcoResolveEncodedSuccList(state, rootFuncList, funcIdx, nodes[n].succList, 0, nodes[n].succSize,
                                           hubStack, hubStackTop, succTaskIdListCoreList, succTaskIdListSizeCoreList);
            }
            TraceEvent(state, curTaskId, EVENT_SUCC_STATIC(succCnt));
            (void)succCnt; // trace 宏关闭时 EVENT_SUCC_STATIC 不求值，抑制 unused-but-set
        } else {
            DrcoResolveStaticSuccNodes(state, rootFuncList, funcIdx, curTaskId, nodes, succNodeIndex, succNodeSize,
                                       hubStack, hubStackTop, succTaskIdListCoreList, succTaskIdListSizeCoreList);
        }

        uint32_t stitchIndex = static_cast<uint32_t>((succInfoBits >> 32) & npu::tile_fwk::DRCO_STITCH_INDEX_MASK);
        if (stitchIndex != 0) {
            // 整链幂分解：先按 2 的幂 + 每 64 一轮 push 子链头入矩阵，
            // 消费核 pop 后按 2 的幂序列（1,2,4,8,16,32）继续接力（每个节点只被解一次）；
            // push 完成后解链头依赖；push 失败则就地解对应子链整段兜底（矩阵写不进则保证前向推进）。
            __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode* chainHead = succStitchList[stitchIndex];
            __gm__ DrcoGlobalStitchNodeMatrix* matrix = DrcoRootFuncListGetStitchNodeMatrix(rootFuncList);

            if (chainHead != nullptr) {
                // 先按 2 的幂序列 push 剩余子链头，最后解链头依赖
                uint32_t rowCursor = 0;
                DrcoStitchNodePushSubChainHeads(state, rootFuncList, matrix, StitchNodeDecodedNext(chainHead),
                                                0xFFFFFFFFu, rowCursor, succTaskIdListCoreList,
                                                succTaskIdListSizeCoreList);
                DrcoResolveStitchNodeTasks(state, rootFuncList, chainHead, curTaskId, hubStack, hubStackTop,
                                           succTaskIdListCoreList, succTaskIdListSizeCoreList);
            }
        }
    }
    DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
}

__aicore__ INLINE static bool MixTaskPush(DrcoEntryState* state, uint32_t succTaskId,
                                          npu::tile_fwk::DrcoMixHubC2VReadyQueue* buf)
{
#if defined(__AIV__)
    (void)state;
    return false;
#else
    if (npu::tile_fwk::DrcoMixHubC2VReadyQueue::Push(buf, succTaskId)) {
        DRCO_LOG(&state->ctx, "MIX C2V push task=%d", (int)succTaskId);
        return true;
    }
    return false;
#endif
}

__aicore__ INLINE static uint32_t ResolveHubMixTask(DrcoEntryState* state,
                                                    __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                    uint32_t hubMixTaskId, npu::tile_fwk::DrcoMixHubC2VReadyQueue* body)
{
    // 区域头（HUB_MIX 型）：命中即开门，跳过 hub 解析
    if (DrcoMiddleQueueRegionHeadTryOpen(state, rootFuncList, hubMixTaskId)) {
        return static_cast<uint32_t>(AICORE_TASK_INIT);
    }
    uint32_t funcIdx = npu::tile_fwk::FuncID(hubMixTaskId);
    uint32_t opIdx = npu::tile_fwk::TaskID(hubMixTaskId);
    __gm__ npu::tile_fwk::DrcoRootFuncData* rootFuncData = DrcoGetCurrentRootFuncData(state, funcIdx);
    __gm__ void* programBase = state->programBase;
    uint64_t funcInfoAndStaticDataOffset = rootFuncData->funcInfoAndStaticDataOffset;
    __gm__ npu::tile_fwk::DevAscendFunctionOperationSuccInfo* mixSuccInfoList = DrcoGetCurrentRootFuncSuccInfoList(
        programBase, static_cast<uint32_t>(funcInfoAndStaticDataOffset & 0x3FFFFFFFULL));
    __gm__ DynFuncData* funcData = &state->ctx.cachedDevTasks[state->ctx.curLeafTaskParallelIdx].funcDataList[funcIdx];
    __gm__ int* mixCceBinaryIndexList = funcData->cceBinaryIndexList;
    uint32_t aicTaskId = static_cast<uint32_t>(AICORE_TASK_INIT);

    // HUB_MIX 生产者永不分块散射：其节点为容量上限顺序追加的混合节点，逐节点逐条目派发
    // （条目语义与平铺时代一致：pair bit 被 TASKID_TASK_MASK 掩掉）
    uint64_t mixSuccInfoBits = mixSuccInfoList[opIdx].staticIndexSizeAndStitchIndex;
    uint16_t mixNodeIndex = static_cast<uint16_t>(mixSuccInfoBits);
    uint16_t mixNodeSize = static_cast<uint16_t>(mixSuccInfoBits >> 16);
    __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* mixNodes = DrcoGetCurrentRootFuncStaticSuccNodeList(
                                                                          programBase, funcInfoAndStaticDataOffset) +
                                                                      mixNodeIndex;
    for (uint16_t n = 0; n < mixNodeSize; n++) {
        __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode* node = &mixNodes[n];
        for (uint32_t j = 0; j < node->succSize; j++) {
            uint32_t mixSuccEncoded = node->succList[j];
            uint32_t mixSuccOpIdx = mixSuccEncoded & TASKID_TASK_MASK;
            uint32_t mixSuccCoreType = (mixSuccEncoded >> npu::tile_fwk::TASKID_DRCO_CT_SHIFT) &
                                       npu::tile_fwk::TASKID_DRCO_CT_MASK;
            uint32_t mixSuccTaskId = npu::tile_fwk::MakeDrcoTaskId(funcIdx, mixSuccOpIdx, mixSuccCoreType);
            DRCO_LOG(&state->ctx, "MIX resolve taskId=%u", mixSuccTaskId);
            if (mixSuccCoreType == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIC)) {
                aicTaskId = mixSuccTaskId;
            } else {
                // 正常路径下环不会满（每次派发每环至多压 1 个、配对 AIV 在 fetch 循环持续 Pop）；
                // 万一暂满则自旋等待：环内有未执行任务时 AIV finish flag 必未置位、配对 AIV 不会退出
                // fetch 循环，等待必有进展；超时 Trap 兜底，杜绝静默丢任务导致的挂死
                npu::tile_fwk::DrcoMixHubC2VReadyQueue* dstAddr = body +
                                                                  state->ctx
                                                                      .cachedDevTasks[state->ctx.curLeafTaskParallelIdx]
                                                                      .cceBinary[mixCceBinaryIndexList[mixSuccOpIdx]]
                                                                      .wrapVecId;
                uint64_t pushStart = get_sys_cnt();
                while (!MixTaskPush(state, mixSuccTaskId, dstAddr)) {
                    if (get_sys_cnt() - pushStart > AICORE_LEAF_TASK_RUN_TIMEOUT) {
                        Trap();
                    }
                    DrcoBusyBackOff();
                }
            }
        }
    }
    return aicTaskId;
}

INLINE __gm__ DrcoLocalReadyQueue* TryGetOtherLocalQue(__gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                       uint32_t coreType, uint32_t groupIdx)
{
    for (uint32_t g = 1; g < npu::tile_fwk::NUM_LOCAL_GROUPS; g++) {
        uint32_t victim = (groupIdx + g) % npu::tile_fwk::NUM_LOCAL_GROUPS;
        __gm__ DrcoLocalReadyQueue* que = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, coreType, victim);
        if (que == nullptr) {
            continue;
        }
        uint32_t head = DrcoAtomicLoad(&que->head);
        uint32_t tail = DrcoAtomicLoad(&que->tail);
        if (head < tail) {
            return que;
        }
    }
    return nullptr;
}

// 消费核 pop 本行 stitch 节点：原子 exch(value -> 0) 读清一体即独占（每槽仅本行主一个
// popper、push 侧只 CAS 空槽，读清无需二次校验）。slot 为 u32（编码见 aikernel_device_task.h）：stitch 链
// 节点按 2 的幂子链头接力；静态后继节点按 funcId 索引 funcData、节点区基址 + 下标还原
// 节点后整节点解依赖（无子链语义，不走 tree relay）
INLINE void DrcoStitchNodeMatrixPopResolve(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                           __gm__ DrcoGlobalStitchNodeMatrix* matrix, uint32_t rowIdx)
{
    // 提前判断（按需走）：stitch 矩阵 COL_SIZE=1（单槽即整行），本核行槽空即无待解节点，直接
    // 返回——省下方批数组清零与 DrcoFlushBatchTasks 空转（fetch 循环每轮无条件调用本函数）。
    // 空槽探测用普通 dcci 读（非 RMW）：本行仅本行主一个 popper、push 侧只 CAS 空槽，读到 0
    // 可安全跳过（探测后新 push 由下一轮捕获）
    if (DrcoGmLoad(&matrix->stitchNodeList[rowIdx].slot[0]) == 0) {
        return;
    }
    uint32_t succTaskIdListCoreList[npu::tile_fwk::DRCO_QUEUE_MAX][BATCH_PUSH_BUF_SIZE];
    uint32_t succTaskIdListSizeCoreList[npu::tile_fwk::DRCO_QUEUE_MAX] = {0};
    int32_t dummyHubStackTop = -1; // pop 的节点不含 HUB（HUB 节点从不入 matrix），仅占位
    for (uint32_t col = 0; col < DrcoGlobalStitchNodeMatrix::COL_SIZE; col++) {
        __gm__ uint32_t* slot = &matrix->stitchNodeList[rowIdx].slot[col];
        // 入口已判 slot[0] 非空（COL_SIZE=1 即整行），此处不再逐槽 dcci 探测
        uint32_t slotVal = DrcoAtomicExchToU32(slot, 0);
        if (slotVal == 0) {
            continue;
        }
        if ((slotVal & npu::tile_fwk::DRCO_STITCH_SLOT_STATIC_SUCC_FLAG) != 0) {
            uint32_t funcId = (slotVal & npu::tile_fwk::DRCO_STITCH_SLOT_FUNCID_MASK) >>
                              npu::tile_fwk::DRCO_STITCH_SLOT_FUNCID_SHIFT;
            uint32_t nodeIndex = slotVal & npu::tile_fwk::DRCO_STITCH_SLOT_NODE_INDEX_MASK;
            // 节点 = 生产 func 节点区基址 + 下标（slot 存 per-func 下标而非字节偏移，
            // u32 位宽内与 funcId/标志共存）
            __gm__ npu::tile_fwk::DrcoRootFuncData* succRootFuncData = DrcoGetCurrentRootFuncData(state, funcId);
            __gm__ npu::tile_fwk::DevAscendFunctionStaticSuccNode*
                node = DrcoGetCurrentRootFuncStaticSuccNodeList(state->programBase,
                                                                succRootFuncData->funcInfoAndStaticDataOffset) +
                       nodeIndex;
            DrcoResolveStaticSuccNodeTasks(state, rootFuncList, funcId, node, nullptr, dummyHubStackTop,
                                           succTaskIdListCoreList, succTaskIdListSizeCoreList);
            continue;
        }
        __gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode*
            node = (__gm__ npu::tile_fwk::DevAscendFunctionDuppedStitchNode*)(rootFuncList->stitchNodeBase + slotVal);
        // 先按 2 的幂序列 push 剩余子链头，最后解本地首节点依赖
        uint32_t remain = StitchNodeRemainingCount(node);
        uint32_t rowCursor = 0;
        DrcoStitchNodePushSubChainHeads(state, rootFuncList, matrix, StitchNodeDecodedNext(node), remain, rowCursor,
                                        succTaskIdListCoreList, succTaskIdListSizeCoreList);
        DrcoStitchNodeTasksResolveCore(state, rootFuncList, node, succTaskIdListCoreList, succTaskIdListSizeCoreList);
    }
    DrcoFlushBatchTasks(state, rootFuncList, succTaskIdListCoreList, succTaskIdListSizeCoreList);
}

INLINE void DrcoDynFuncDataListFetchResolveStitchNodeMatrix(DrcoEntryState* state,
                                                            __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                            uint32_t blockIdx)
{
    __gm__ DrcoGlobalStitchNodeMatrix* stitchNodeMatrix = DrcoRootFuncListGetStitchNodeMatrix(rootFuncList);
    DrcoStitchNodeMatrixPopResolve(state, rootFuncList, stitchNodeMatrix, blockIdx);
}

// 消费本核行上的兜底 hub 任务（hubStack 溢出/stitch 类型越界改投）：与 DrcoStitchNodeMatrixPopResolve
// 相同的原子 exch 抢占（exch(val -> 0) 读清一体即独占），pop 后经 DrcoResolveDepend 就地
// 解依赖（不执行不计数）；fetch 循环每轮无条件调用（帮忙模式含）——矩阵内任务必被消费，
// 消除"溢出 hub 落本类型队列 + 本类型 flag 已置位核跳过 pop"的 stranded 死锁死角
INLINE void DrcoHubTaskMatrixPopResolve(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                        uint32_t rowIdx)
{
    __gm__ npu::tile_fwk::DrcoGlobalHubTaskMatrix* matrix = rootFuncList->hubTaskMatrix;
    if (matrix == nullptr) {
        return;
    }
    // 提前判断（按需走）：hub 矩阵 COL_SIZE=1（单槽即整行），本核行槽空即无需 pop resolve，
    // 直接返回（fetch 循环每轮无条件调用本函数）；空槽探测为普通 dcci 读，非空才走 u32 exch
    if (DrcoGmLoad(&matrix->hubTaskList[rowIdx].slot[0]) == 0) {
        return;
    }
    for (uint32_t col = 0; col < npu::tile_fwk::DrcoGlobalHubTaskMatrix::COL_SIZE; col++) {
        __gm__ uint32_t* slot = &matrix->hubTaskList[rowIdx].slot[col];
        // 入口已判 slot[0] 非空（COL_SIZE=1 即整行），此处不再逐槽 dcci 探测
        uint32_t encoded = DrcoAtomicExchToU32(slot, 0);
        if (encoded == 0) {
            continue;
        }
        uint32_t taskId = DRCO_DECODE_TASK(encoded);
        DrcoResolveDepend(state, rootFuncList, &taskId);
    }
}

INLINE bool DrcoDynFuncDataListFetchTaskMixHubC2VReadyQueue(
    DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList, uint32_t& resultCoreType,
    uint32_t resultTaskIdList[npu::tile_fwk::DRCO_MATRIX_COL_MAX], uint32_t& resultTaskIdCount, uint32_t blockIdx)
{
    if (IS_AIV) {
        uint32_t bodyTaskId = 0;
        if (npu::tile_fwk::DrcoMixHubC2VReadyQueue::Pop(state->mixHubC2VReadyQueue, bodyTaskId)) {
            resultTaskIdList[0] = bodyTaskId;
            resultTaskIdCount = 1;
            resultCoreType = npu::tile_fwk::DRCO_QUEUE_AIV;
            return true;
        }
    }
    return false;
}

INLINE bool DrcoDynFuncDataListFetchTaskLocalReadyMatrix(DrcoEntryState* state,
                                                         __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                         uint32_t& resultCoreType,
                                                         uint32_t resultTaskIdList[npu::tile_fwk::DRCO_MATRIX_COL_MAX],
                                                         uint32_t& resultTaskIdCount,
                                                         __gm__ DrcoLocalReadyMatrix* localReadyMatrix, uint32_t rowIdx)
{
    uint32_t count = DrcoLocalReadyMatrixPopRowTasks(localReadyMatrix, rowIdx, resultTaskIdList, DRCO_FETCH_BATCH_SIZE);
    if (count > 0) {
        resultTaskIdCount = count;
        resultCoreType = DRCO_CORE_TYPE;
        return true;
    }
    return false;
}

// queueCoreType 参数化：本类型行（AIC/AIV）与 MIX 行共用同一取数逻辑，命中时 resultCoreType = queueCoreType；
// 仅取本组队列，不跨组窃取（push 始于推送核本组、活跃区间内每组均有本组消费者，本组队列必被消费）
INLINE bool DrcoDynFuncDataListFetchTaskLocalReadyQueue([[maybe_unused]] DrcoEntryState* state,
                                                        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                                        uint32_t queueCoreType, uint32_t& resultCoreType,
                                                        uint32_t resultTaskIdList[npu::tile_fwk::DRCO_MATRIX_COL_MAX],
                                                        uint32_t& resultTaskIdCount, uint32_t groupIdx)
{
    __gm__ DrcoLocalReadyQueue* localReadyQueue = DrcoRootFuncListGetLocalReadyQueue(rootFuncList, queueCoreType,
                                                                                     groupIdx);
    // 冗余 dcci 删除：GetFirstTask 现已逐次对 head/tail 做 dcci 读，无需再额外失效一次
    uint32_t taskId = static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT);
    while (taskId == static_cast<uint32_t>(AICORE_TASK_FETCH_CONFLICT)) {
        taskId = DrcoLocalReadyQueueGetFirstTask(localReadyQueue);
    }
    if (taskId != static_cast<uint32_t>(AICORE_TASK_NO_INCOME)) {
        resultTaskIdList[0] = taskId;
        resultTaskIdCount = 1;
        resultCoreType = queueCoreType;
        return true;
    }

    return false;
}

// mixhub 与 AIC/AIV 同模型消费：从 MIX 矩阵取任务（行号按 AIC 核的类型内编号，消费者
// 为 AIC），先 pop 合并 matrix 自己的行，再取本组 MIX queue；取出后不执行
// leaf，仅由 ExecDrcoReadyQueueTaskOnce 经 ResolveHubMixTask 解依赖派发（AIC 后继就地执行、
// AIV 后继 C2V 投递配对 AIV）
INLINE bool DrcoDynFuncDataListFetchTaskMixHub(DrcoEntryState* state,
                                               __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                               uint32_t& resultCoreType,
                                               uint32_t resultTaskIdList[npu::tile_fwk::DRCO_MATRIX_COL_MAX],
                                               uint32_t& resultTaskIdCount, uint32_t groupIdx, uint32_t rowIdx)
{
    if (IS_AIC) {
        __gm__ DrcoLocalReadyMatrix* mixMatrix = DrcoRootFuncListGetLocalReadyMatrix(rootFuncList,
                                                                                     npu::tile_fwk::DRCO_QUEUE_MIX);
        if (mixMatrix != nullptr && mixMatrix->validCoreNum > static_cast<uint32_t>(rowIdx)) {
            uint32_t count = DrcoLocalReadyMatrixPopRowTasks(mixMatrix, rowIdx, resultTaskIdList,
                                                             DRCO_FETCH_BATCH_SIZE);
            if (count > 0) {
                resultTaskIdCount = count;
                resultCoreType = npu::tile_fwk::DRCO_QUEUE_MIX;
                return true;
            }
        }
        return DrcoDynFuncDataListFetchTaskLocalReadyQueue(state, rootFuncList, npu::tile_fwk::DRCO_QUEUE_MIX,
                                                           resultCoreType, resultTaskIdList, resultTaskIdCount,
                                                           groupIdx);
    }
    return false;
}

INLINE bool DrcyDynFuncDataListFetchSelfQueue(DrcoEntryState* state, uint32_t& resultCoreType,
                                              uint32_t resultTaskIdList[npu::tile_fwk::DRCO_MATRIX_COL_MAX],
                                              uint32_t& resultTaskIdCount)
{
    // selfReadyQueue 优先消费：resolve 收尾时直通到本核 selfReadyQueue 的同类型就绪后继，
    // fetch 开头一次性全部取走执行；消费即清空，直通任务照常走 TaskOnce 执行与计数，
    // 不绕过 finish 语义（selfReadyQueue 容量 <= resultTaskIdList 容量，可整批取出）
    if (state->selfReadyQueueSize > 0) {
        uint32_t count = state->selfReadyQueueSize;
        for (uint32_t i = 0; i < count; i++) {
            TraceEvent(state, state->selfReadyQueue[i], EVENT_POP_SELF_READY_QUEUE());
            resultTaskIdList[i] = state->selfReadyQueue[i];
        }
        state->selfReadyQueueSize = 0;
        resultCoreType = DRCO_CORE_TYPE;
        resultTaskIdCount = count;
        return true;
    }
    return false;
}

// 分片完成判定兜底：Σ分片 >= 期望动态计数（size - 静态预累加）时置本类型完成 flag。
// 分片计数下，最后一个 add 的核在其它核 add 尚未全局可见时可能读漏，若此后无人再读即漏判挂死；
// fetch 空转路径每轮调用本函数补判，保证任一空转核读到全局总和即置位（不会漏判）
INLINE void DrcoTryFinishByShardCount(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    __gm__ uint32_t* flag = &rootFuncList->devTaskFinishFlagList.flag[DRCO_CORE_TYPE].devTaskFinishFlag;
    if (DrcoGmLoad(flag) != 0) {
        return;
    }
    uint32_t total = 0;
    for (uint32_t s = 0; s < npu::tile_fwk::DRCO_COUNT_SHARD_NUM; s++) {
        total += DrcoGmLoad(&rootFuncList->devTaskCountList.shardExecCount[DRCO_CORE_TYPE][s].value);
    }
    if (total >= state->drcoCountDynExpect) {
        DrcoGmStore(flag, uint32_t(1));
    }
}

INLINE uint32_t DrcoDynFuncDataListFetchTask(DrcoEntryState* state,
                                             __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                             uint32_t& resultCoreType,
                                             uint32_t resultTaskIdList[npu::tile_fwk::DRCO_MATRIX_COL_MAX])
{
    TraceEvent(state, 0xffffffff, EVENT_FETCH_START());
    BlockDesc blockDesc = state->blockDesc;
    uint32_t blockIdx = BlockDescBlockIdx(blockDesc);
    uint32_t typedBlockIdx = BlockDescTypedBlockIdx(blockDesc);
    uint32_t groupIdx = typedBlockIdx / npu::tile_fwk::LOCAL_GROUP_SIZE; // local queue 仍按分组

    __gm__ DrcoLocalReadyMatrix* localReadyMatrix = DrcoRootFuncListGetLocalReadyMatrix(rootFuncList, DRCO_CORE_TYPE);
    // 合并矩阵：行 = 类型内本地编号本身
    uint32_t rowIdx = typedBlockIdx;
    // finishFlag 每类型共享一个：本类型全部核经 DrcoGmLoad（含 dcci 失效读）轮询同一 flag
    // （ssbuf 协议下 V 端改读 ssbuf 完成镜像，不走这两个 GM flag 指针）
    __gm__ uint32_t* devTaskFinishFlag = &rootFuncList->devTaskFinishFlagList.flag[DRCO_CORE_TYPE].devTaskFinishFlag;
    // 对方类型完成 flag。
    // 全局退出 = 两类型 flag 均置位；本类型先完成时进入帮忙模式（只扫共享 stitch 矩阵，
    // 跳过本类型 leaf 矩阵/队列 pop——必然空，省 CAS 竞争），避免对侧链节点落在已退出核的行无人 pop
    constexpr uint32_t DRCO_OTHER_QUEUE_TYPE = (DRCO_CORE_TYPE == static_cast<uint32_t>(npu::tile_fwk::CoreType::AIC)) ?
                                                   npu::tile_fwk::DRCO_QUEUE_AIV :
                                                   npu::tile_fwk::DRCO_QUEUE_AIC;
    __gm__ uint32_t* otherTaskFinishFlag = &rootFuncList->devTaskFinishFlagList.flag[DRCO_OTHER_QUEUE_TYPE]
                                                .devTaskFinishFlag;
    uint32_t resultTaskIdCount = 0;

    uint64_t t0 = get_sys_cnt();
    while (true) {
        // middleQueue 门探测（每轮第一句，任何模式含 help 模式都必探）：开门代加后 own flag
        // 可能在本核消费前置位（flag ⇒ 门已开，不再蕴含"已消费"），help 模式（own=1,
        // other=0）若跳过探测，成员被搁置到对侧完成才消费——成员含对侧后继时互等死锁；
        // 探测前置同时把发现延迟压到最短（先于 flag 双读与各矩阵扫描）
        if (DrcoMiddleQueueTryProbeGate(state, resultTaskIdCount)) {
            break;
        }
        // 全局退出 = 本类型 + 对方类型 flag 均置位（leaf 计数含 mix 任务，两类型齐全即全部完成）；
        // 单侧完成不退出——继续循环扫共享 stitch 矩阵帮忙对侧，防对侧链节点落在己方行无人 pop。
        // 双检：顶部探针与 flag 读之间门可能新开+账凑齐（写序 0xff → add → flag），退出前
        // 再探一次未消费槽必命中，防成员核带任务退出（静默丢任务）
        // V 端一轮一次读 ssbuf 完成镜像（FINISH_ALL / FINISH_VECTOR 位复用，无 GM dcci）；
        // C 端 dcci 读 GM 双 flag
        uint32_t ssbufFinishState = IS_AIV ? npu::tile_fwk::DrcoWarpExecuteCount::FinishStateLoad(
                                                 &state->ssbuf->warpExecuteCount) :
                                             0;
        // C 端本轮 own flag 缓存一次（下方 ownTypeFinished 复用），省一次 GM dcci 重复读
        uint32_t ownFinishFlag = IS_AIV ? 0 : DrcoGmLoad(devTaskFinishFlag);
        bool allTaskFinished = IS_AIV ? ((ssbufFinishState & npu::tile_fwk::DrcoWarpExecuteCount::FINISH_ALL) != 0) :
                                        (ownFinishFlag != 0 && DrcoGmLoad(otherTaskFinishFlag) != 0);
        if (allTaskFinished) {
            if (DrcoMiddleQueueTryProbeGate(state, resultTaskIdCount)) {
                break;
            }
            if (IS_AIC) {
                // C 端：双 flag 确认无待消费任务后发布全局退出镜像，放行本组 V
                npu::tile_fwk::DrcoWarpExecuteCount::FinishPublish(&state->ssbuf->warpExecuteCount,
                                                                   npu::tile_fwk::DrcoWarpExecuteCount::FINISH_ALL);
            }
            resultTaskIdCount = static_cast<uint32_t>(AICORE_TASK_ALL_FINISH);
            break;
        }

        if (get_sys_cnt() - t0 > AICORE_LEAF_TASK_RUN_TIMEOUT) {
            Trap();
        }

        // 直通 selfReadyQueue 优先消费（每轮循环开头查）：stitch 扫描/展开可能在【本轮循环内】
        // 经 FlushBatchTasks 写入 selfQueue（同类型后继直通路径），若只在函数入口查一次，写入后
        // 本函数空转期间槽内任务永久饿死（实测卡死教训：检查必须在循环内每轮开头）
        if (DrcyDynFuncDataListFetchSelfQueue(state, resultCoreType, resultTaskIdList, resultTaskIdCount)) {
            break;
        }

        // stitch 矩阵扫描：仅在图中存在 stitch 节点时开启（hasStitchWork，运行时判定；
        // DRCO_OPT_STITCH_GATE 为构建开关，置 0 恢复无条件调用）
        if (state->hasStitchNode) {
            DrcoDynFuncDataListFetchResolveStitchNodeMatrix(state, rootFuncList, blockIdx);
        }
        // 兜底 hub 矩阵扫描：由 push 侧首次成功改投时置位的共享标记驱动（state 缓存；未见
        // 置位时每轮补读一次共享标记，置位后仅剩本行空槽探测，帮忙模式含）。pop 后就地解
        // 依赖，pop 出的任务不进 taskIdList（不执行不计数），own-type 批因此恒为纯 leaf 批
        DrcoHubTaskMatrixPopResolve(state, rootFuncList, blockIdx);

        // 本类型 leaf 全部认领/执行完（flag 置位）后跳过本类型 leaf 矩阵/队列/mixhub 消费
        // （必然空，省 CAS 竞争）。V 端门控复用本轮镜像值，无 GM dcci。
        // 探测前置：本核行/本组队列命中概率最高，先于共享矩阵扫描可缩短 push→pop 相位；
        // 全未命中时本轮继续落在下方共享 stitch/hub 扫描（含帮忙模式），不丢服务
        bool ownTypeFinished = IS_AIV ? ((ssbufFinishState & npu::tile_fwk::DrcoWarpExecuteCount::FINISH_VECTOR) != 0) :
                                        ownFinishFlag != 0;
        if (!ownTypeFinished) {
            // MIX 分支（C2V 环 + MIX 矩阵行扫 + MIX 队列）仅在本图存在 HUB_MIX 后继时才可能
            // 非空。DRCO_FLAG_HAS_MIX 由 host 在 kernel 编译时按图注入：有=1（保留扫描，运行时
            // 判定）、无=0（整段编译裁剪，矩阵/环恒无生产者），并保留一致性校验
#if DRCO_FLAG_HAS_MIX
            if (DrcoDynFuncDataListFetchTaskMixHubC2VReadyQueue(state, rootFuncList, resultCoreType, resultTaskIdList,
                                                                resultTaskIdCount, blockIdx)) {
                break;
            }
            if (DrcoDynFuncDataListFetchTaskMixHub(state, rootFuncList, resultCoreType, resultTaskIdList,
                                                   resultTaskIdCount, groupIdx, rowIdx)) {
                break;
            }
#endif
            if (DrcoDynFuncDataListFetchTaskLocalReadyMatrix(state, rootFuncList, resultCoreType, resultTaskIdList,
                                                             resultTaskIdCount, localReadyMatrix, rowIdx)) {
                break;
            }
            if (DrcoDynFuncDataListFetchTaskLocalReadyQueue(state, rootFuncList, DRCO_CORE_TYPE, resultCoreType,
                                                            resultTaskIdList, resultTaskIdCount, groupIdx)) {
                break;
            }
        } // 本类型未完成（帮忙模式跳过 leaf 消费）

        // 本轮未取到任务：分片完成判定兜底，避免分片计数读漏导致完成 flag 永不置位而挂死
        // （V 端无判定职责；C 端追加聚合，保证 help 模式下持续推进）
        if (IS_AIC) {
            DrcoTryFinishByShardCount(state, rootFuncList);
            DrcoAggregateVectorCount(state, rootFuncList);
        }
    }
    TraceEvent(state, 0xffffffff, EVENT_FETCH_END());
    return resultTaskIdCount;
}

INLINE void InitDrcoEntry(DrcoEntryState* state, int64_t cfgdata)
{
    auto devArgs = (DeviceArgs*)cfgdata;
    uint64_t start = get_sys_cnt();
#if defined(__AIV__) && defined(__MIX__)
    uint32_t blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    uint32_t blockIdx = get_block_idx();
#endif
    uint32_t typedBlockIdx = DrcoGetCoreTypedIdx(blockIdx, devArgs->nrValidAic);
    state->blockDesc = BlockDescCreate(blockIdx, typedBlockIdx, devArgs->nrValidAic, devArgs->nrValidAic * 2);
    state->args = (__gm__ KernelArgs*)(devArgs->sharedBuffer +
                                       BlockDescBlockIdx(state->blockDesc) * SHARED_BUFFER_SIZE);
    __gm__ FedProfPerCoreProf* metric = (__gm__ FedProfPerCoreProf*)(devArgs->fedProfAddrList[FED_PROF_PER_CORE_PROF] +
                                                                     BlockDescBlockIdx(state->blockDesc) *
                                                                         FED_PROF_PER_CORE_PROF_ALIGNED_SIZE);
    state->metric = metric;
    state->ctx.args = state->args;
    state->ctx.blockIdx = BlockDescBlockIdx(state->blockDesc);
    state->ctx.perCoreProf = metric;
    __gm__ DevDfxArgs* devDfxAddr = (__gm__ DevDfxArgs*)devArgs->devDfxArgAddr;
    state->ctx.devTaskProfStatus.profEnabled = devDfxAddr->isOpenPerfTrace != 0;
    state->ctx.profLevel = devDfxAddr->profLevel;
    state->ctx.aicCoreNum = devArgs->nrValidAic;
    // 各 coreType 活跃组区间是本 kernel 生命周期内的不变量，入口一次算好供全部热路径查表
    for (uint32_t ct = 0; ct < npu::tile_fwk::DRCO_QUEUE_MAX; ct++) {
        GetDrcoActiveGroupRange(ct, state->ctx.aicCoreNum, state->ctx.drcoGroupBeg[ct], state->ctx.drcoGroupEnd[ct]);
    }

    uint8_t aicoreLogLevel = static_cast<uint8_t>(AicoreLogLevel::NONE);
#if ENABLE_AICORE_PRINT
    if (devDfxAddr->logLevel >= 0) {
        aicoreLogLevel = static_cast<uint8_t>(devDfxAddr->logLevel);
    }
#endif
    if (state->ctx.devTaskProfStatus.profEnabled && metric->status.currLaunchRound < FED_PROF_DEV_TASK_MAX_COUNT) {
        uint64_t round = metric->status.currLaunchRound;
        state->ctx.devTaskProfStatus.devTaskProf = &(metric->aicoreDevTaskInfo[round]);
        FedProfDevTaskProfRecord(INVALID_DEV_TASK_ID, state->ctx.devTaskProfStatus.devTaskProf, PERF_TRACE_CORE_BEGIN,
                                 start);
    }

    set_mask_norm();
    state->lastMixResourceType = static_cast<uint8_t>(MixResourceType::MIX_UNKNOWN);

    FedProfDevTaskProfRecord(INVALID_DEV_TASK_ID, state->ctx.devTaskProfStatus.devTaskProf, PERF_TRACE_CORE_INIT);

    InitCtx(&state->ctx, metric, nullptr, aicoreLogLevel);

    state->runtimeDataRingBufferHeadData = (__gm__ npu::tile_fwk::RuntimeDataRingBufferHeadData*)
                                               devArgs->runtimeDataRingBufferAddr;
    DRCO_DCCI_SINGLE_CACHE_LINE(state->runtimeDataRingBufferHeadData);
    state->base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
        state->runtimeDataRingBufferHeadData);
    DRCO_DCCI_SINGLE_CACHE_LINE(state->base);
    state->deviceTaskReadyQueue = DrcoGmLoad(&state->base->drcoDeviceTaskReadyQueue);
}

INLINE __gm__ DrcoDeviceTask* GetDrcoDeviceTask(DrcoEntryState* state, bool& isFirstTask, uint64_t t0,
                                                uint64_t& loopCount, bool& warningSet)
{
    if (BlockDescBlockIdx(state->blockDesc) == 0) {
        if (!isFirstTask) {
            uint32_t oldHead = DrcoAtomicLoad(&state->deviceTaskReadyQueue->head);
            DrcoAtomicCasToU32(&state->deviceTaskReadyQueue->head, oldHead, oldHead + 1);
        }
        isFirstTask = false;
        __gm__ DrcoDeviceTask* deviceTask = nullptr;
        while (true) {
            DRCO_LEADER_TIMEOUT_CHECK(t0, loopCount, AICORE_LEAF_TASK_RUN_TIMEOUT, STAGE_RUN_LEAFTASK_TIMEOUT);
            if (state->deviceTaskReadyQueue == nullptr) {
                DRCO_DCCI_SINGLE_CACHE_LINE(state->runtimeDataRingBufferHeadData);
                state->base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
                    state->runtimeDataRingBufferHeadData);
                DRCO_DCCI_SINGLE_CACHE_LINE(state->base);
                state->deviceTaskReadyQueue = DrcoGmLoad(&state->base->drcoDeviceTaskReadyQueue);
                if (state->deviceTaskReadyQueue == nullptr) {
                    DrcoBusyBackOff();
                    continue;
                }
            }
            deviceTask = GetCurrentDeviceTask(state->deviceTaskReadyQueue);
            if (deviceTask == nullptr) {
                uint32_t qHead = DrcoAtomicLoad(&state->deviceTaskReadyQueue->head);
                uint32_t qTail = DrcoGmLoad(&state->deviceTaskReadyQueue->tail);
                if (qHead < qTail) {
                    __gm__ DrcoDeviceTask*
                        elem = &state->deviceTaskReadyQueue
                                    ->dynFuncDataListList[qHead % npu::tile_fwk::DEVICE_TASK_QUEUE_SIZE];
                    if (DrcoGmLoad(&elem->dynFuncDataList) == nullptr) {
                        break;
                    }
                }
                DrcoBusyBackOff();
                continue;
            }
            break;
        }
        SyncAllMix();
        return deviceTask;
    }

    SyncAllMix();
    if (state->deviceTaskReadyQueue == nullptr) {
        DRCO_DCCI_SINGLE_CACHE_LINE(state->runtimeDataRingBufferHeadData);
        state->base = npu::tile_fwk::RuntimeDataRingBufferHeadData::GetRuntimeDataCurrent(
            state->runtimeDataRingBufferHeadData);
        DRCO_DCCI_SINGLE_CACHE_LINE(state->base);
        state->deviceTaskReadyQueue = DrcoGmLoad(&state->base->drcoDeviceTaskReadyQueue);
    }
    __gm__ DrcoDeviceTask* deviceTask = GetCurrentDeviceTask(state->deviceTaskReadyQueue);
    return deviceTask;
}

INLINE void ExecLeafFunction(DrcoEntryState* state, uint32_t taskId)
{
    TraceEvent(state, taskId, EVENT_LEAF_START());

    ExecCoreFunctionKernel(&state->ctx, taskId, state->lastMixResourceType);

    TraceEvent(state, taskId, EVENT_LEAF_END());
}

INLINE void PerfDevTaskFirstLeafTask(DrcoEntryState* state)
{
    if (!state->isExectedLeafTask) {
        FedProfDevTaskProfRecord(state->ctx.SeqNo(), state->ctx.devTaskProfStatus.devTaskProf,
                                 PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK);
        state->isExectedLeafTask = true;
    }
}

// middleQueue 成员标记：leaf 成员（执行后 resolve 区域外后继）与 hub 成员（只 resolve 不执行）
INLINE bool DrcoPerCoreTaskMiddleQueueMemberHub(uint32_t taskId)
{
    return ((taskId >> npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_SHIFT) & npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_MASK) ==
           npu::tile_fwk::DRCO_SUCC_SPECIAL_MIDDLE_QUEUE_MEMBER_HUB;
}

// MQ 成员 resolve 跳过标记（host 撒核入队时按"唯一后继为 dummyEnding 且无 stitch、未内存
// 复用缝合"同 DrcoRefreshSpecialTaskId 判据预编码）：命中者 resolve 到的全是 dummy 边
// （设备侧本就跳过递减），无任何下游可解，消费侧直接跳过 resolve
INLINE bool DrcoPerCoreTaskMiddleQueueMemberSkipResolve(uint32_t taskId)
{
    return ((taskId >> npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_SHIFT) & npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_MASK) ==
           npu::tile_fwk::DRCO_SUCC_SPECIAL_MIDDLE_QUEUE_MEMBER_SKIP;
}

INLINE bool DrcoPerCoreTaskMiddleQueueMember(uint32_t taskId)
{
    return ((taskId >> npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_SHIFT) & npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_MASK) ==
           npu::tile_fwk::DRCO_SUCC_SPECIAL_MIDDLE_QUEUE_MEMBER;
}

// per-core queue 任务是否带「唯一后继为 dummyEnding 可跳过 resolve」标记（special 字段命中 SKIP_RESOLVE）
INLINE bool DrcoPerCoreTaskSkipResolve(uint32_t taskId)
{
    return ((taskId >> npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_SHIFT) & npu::tile_fwk::TASKID_DRCO_SUCC_SPECIAL_MASK) ==
           npu::tile_fwk::DRCO_SUCC_SPECIAL_PERCORE_SKIP_RESOLVE;
}

// 门判定：一解多直通成员恒为单前驱（分量拓扑序），门只由头 predCount==0xff 决定
INLINE bool DrcoMiddleQueueGateOpen(DrcoEntryState* state, uint32_t s)
{
    return DrcoGmLoad(DrcoTaskPredCountAddr(state, state->middleQueue.headTaskId[s])) ==
           npu::tile_fwk::DRCO_MIDDLE_QUEUE_GATE_OPEN;
}

// middleQueue 消费
INLINE void DrcoTryMiddleQueueConsume(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    for (uint32_t s = 0; s < npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM; s++) {
        if (state->middleQueue.consumed[s]) {
            continue;
        }
        __gm__ npu::tile_fwk::PerCoreMiddleQueue* middleQueue = state->middleQueue.queue[s];
        if (middleQueue == nullptr) {
            continue;
        }
        uint32_t queueSize = state->middleQueue.queueSize[s];
        if (queueSize == 0) {
            continue;
        }
        // probe 刚验过的槽不再复读门（0xff 粘性 + consumed 防重入）：16 核同时 DCCI 同一
        // head predCount cacheline 的争用风暴即由此复读引发；未验过的槽（多 region 场景
        // 下 consume 扫到 probe 未覆盖的槽）仍走完整门检查
        if (s != state->middleQueue.verifiedSlot && !DrcoMiddleQueueGateOpen(state, s)) {
            continue;
        }
        // 消费权仅需核本地标记：rootFuncList/middleQueue 由 InitDrcoRootFuncList 按 devTask
        // 独立分配，核本地 middleQueueConsumed 在每 devTask 加载段复位、每 launch 全新
        // ——同 devTask 内不存在"同一队列被重复消费"的场景，无需 GM 仲裁
        state->middleQueue.consumed[s] = true;
        // leaf 计数已由开门核在写 0xff 时一次性代加（DrcoMiddleQueueRegionHeadTryOpen），此处不再逐核记账
        for (uint32_t i = 0; i < queueSize; i++) {
            // 队列条目 = 裸 flatId | MQ 标记（bits26-31）；标记仅用于条目分类，执行/resolve
            // 必须用剥标后的裸 id——tag 位与 ct/parallelIdx 位重叠，带标 id 会使
            // DrcoTaskCoreTypeOf 读出 ct=7（hub 路由）、ParallelIndex 读出 7（越界风险）
            uint32_t taskId = DrcoGmLoadArray(middleQueue->taskList, i);
            bool isHubMember = DrcoPerCoreTaskMiddleQueueMemberHub(taskId);
            bool skipResolve = DrcoPerCoreTaskMiddleQueueMemberSkipResolve(taskId);
            taskId &= TASKID_FROM_CTRL_TOPO_MASK;
            if (isHubMember) {
                DrcoResolveDepend(state, rootFuncList, &taskId);
                continue;
            }
            PerfDevTaskFirstLeafTask(state);
            ExecLeafFunction(state, taskId);
            if (!skipResolve) {
                DrcoResolveDepend(state, rootFuncList, &taskId);
            }
        }
    }
    state->middleQueue.verifiedSlot = 0xFF;
}

INLINE void ExecDrcoPerCoreTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue,
                                 __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    uint32_t perCoreQueueSize = DrcoGmLoad(&perCoreQueue->size);
    uint32_t perCoreQueueHead = DrcoGmLoad(&perCoreQueue->head);
    while (perCoreQueueSize > perCoreQueueHead) {
        uint32_t taskId = DrcoGmLoadArray(perCoreQueue->taskList, perCoreQueueHead);
        PerfDevTaskFirstLeafTask(state);
        ExecLeafFunction(state, taskId);
        // 区域头开门/跳过 resolve 统一由 DrcoResolveDepend 的 hubStack 首查处理
        if (!DrcoPerCoreTaskSkipResolve(taskId)) {
            DrcoResolveDepend(state, rootFuncList, &taskId);
        }
        perCoreQueueHead += 1;
    }
}

// 执行单个就绪任务，恒执行恰好一个 leaf：own-type 批任务即本类型可执行 leaf（hubStack 溢出/
// stitch 类型越界兜底已改投全核共享 hub 矩阵，fetch 无条件 pop 就地解依赖）；MIX 行任务为
// hub_mix 节点，经 ResolveHubMixTask 解析后就地执行其唯一 AIC 后继（编码侧
// ValidateWrapGroupConsistency 断言每个 wrap 组恒为 1 AIC + 1~2 AIV，hub_mix 恒有 AIC 后继）、
// AIV 后继经 C2V 环投递配对 AIV——MIX 批的 taskId 原位覆写为实际执行的 AIC 后继，供调用方
// 统一 resolve。动态完成计数由调用方在执行循环前整批累加（notify before run）：计数到
// size 即本类型所有 leaf 已被认领，末批执行期间其它核可并行退出 fetch 自旋，消除整批
// 执行完才广播的尾部空隙
INLINE void ExecDrcoReadyQueueTaskOnce(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                       [[maybe_unused]] __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue,
                                       uint32_t& taskId, uint32_t outCoreType)
{
#if defined(__MIX__) && defined(__AIC__)
    if (outCoreType == npu::tile_fwk::DRCO_QUEUE_MIX) {
        // mixhub 任务（来自 MIX 行 matrix/queue）不执行 leaf，仅解依赖派发
        uint32_t aicTaskId = ResolveHubMixTask(state, rootFuncList, taskId, state->ssbuf->mixHubC2VReadyQueue);
        if (aicTaskId == static_cast<uint32_t>(AICORE_TASK_INIT)) {
            // 无 AIC 后继：V 后继已在 ResolveHubMixTask 内经 C2V 派发完毕，hub_mix 节点
            // 自身非可执行 leaf（不计入完成计数），无事可做
            return;
        }
        DRCO_LOG(&state->ctx, "MIX exec=%u", aicTaskId);
        taskId = aicTaskId;
    }
#endif
    ExecLeafFunction(state, taskId);
}

INLINE void ExecDrcoReadyQueueTasks(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList,
                                    __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue)
{
    uint32_t taskIdList[npu::tile_fwk::DRCO_MATRIX_COL_MAX];

    uint32_t outCoreType = 0;
    uint32_t taskCount = DrcoDynFuncDataListFetchTask(state, rootFuncList, outCoreType, taskIdList);
    while (taskCount != static_cast<uint32_t>(AICORE_TASK_ALL_FINISH)) {
        if (state->middleQueue.pending) {
            state->middleQueue.pending = false;
            DrcoTryMiddleQueueConsume(state, rootFuncList);
        }
        PerfDevTaskFirstLeafTask(state);
        DrcoNotifyTaskExecutedAdd(state, rootFuncList, taskCount);
        // 区域头开门/跳过 resolve 统一由 DrcoResolveDepend 的 hubStack 首查处理
        for (uint32_t i = 0; i < taskCount; i++) {
            ExecDrcoReadyQueueTaskOnce(state, rootFuncList, perCoreQueue, taskIdList[i], outCoreType);
        }
        DrcoResolveDepend(state, rootFuncList, taskIdList, taskCount);
        taskCount = DrcoDynFuncDataListFetchTask(state, rootFuncList, outCoreType, taskIdList);
    }
}

// middleQueue 相关状态初始化（每 devTask 载入一次）：
//  · 各槽队列指针 + 门探测常量（headTaskId，encode 期不变量）缓存到核本地
//  · 区域头注册表载入核本地副本（DrcoMiddleQueueRegionHeadTryOpen 在任务完成点高频调用，避免读 GM）
INLINE void DrcoMiddleQueueInit(DrcoEntryState* state, __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList)
{
    for (uint32_t s = 0; s < npu::tile_fwk::DRCO_MIDDLE_QUEUE_SLOT_NUM; s++) {
        __gm__ npu::tile_fwk::PerCoreMiddleQueue* mq = DrcoGmLoad(
            &rootFuncList->perCoreMiddleQueueArray[BlockDescBlockIdx(state->blockDesc)][s]);
        state->middleQueue.queue[s] = mq;
        state->middleQueue.consumed[s] = false;
        // headTaskId/size 同 cacheline 相邻：一次 u64 读回双字段，size 供 consume 免冷 dcci
        if (mq != nullptr) {
            uint64_t headAndSize = DrcoGmLoad(reinterpret_cast<__gm__ uint64_t*>(mq));
            state->middleQueue.headTaskId[s] = static_cast<uint32_t>(headAndSize);
            state->middleQueue.queueSize[s] = static_cast<uint32_t>(headAndSize >> 32);
        } else {
            state->middleQueue.headTaskId[s] = 0;
            state->middleQueue.queueSize[s] = 0;
        }
    }
    state->middleQueue.pending = false;
    state->middleQueue.verifiedSlot = 0xFF;
    state->middleQueue.regionHeadCount = DrcoGmLoad(&rootFuncList->middleQueueRegionHeadCount);
    for (uint32_t i = 0;
         i < state->middleQueue.regionHeadCount && i < npu::tile_fwk::DrcoRootFuncList::DRCO_MIDDLE_QUEUE_REGION_MAX;
         i++) {
        state->middleQueue.regionHeadList[i] = DrcoGmLoad(&rootFuncList->middleQueueRegionHeadList[i]);
    }
}

INLINE void KernelEntryDrco(int64_t ffts_addr, int64_t inputs, int64_t outputs, int64_t workspace, int64_t tilingdata,
                            int64_t cfgdata)
{
    UNUSED(ffts_addr);
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(workspace);
    UNUSED(tilingdata);

    DrcoEntryState state = {};
    InitDrcoEntry(&state, cfgdata);
    TraceInit(&state);

    bool isFirstTask = true;
    AICORE_TIMEOUT_CHECK_BEGIN(t0, loop_count);
    while (true) {
        __gm__ DrcoDeviceTask* deviceTask = GetDrcoDeviceTask(&state, isFirstTask, t0, loop_count, warningSet);
        if (deviceTask == nullptr) {
            break;
        }

        UpdateCacheDevTask(&state.ctx, state.ctx.curLeafTaskParallelIdx, (int64_t)deviceTask->dynFuncDataList);
        __gm__ npu::tile_fwk::DrcoRootFuncList* rootFuncList = deviceTask->drcoRootFuncList;
        state.ctx.lastTaskFinishCycle = 0;
        state.isExectedLeafTask = false;
        // size 与静态预累加 staticExecCount 只读且 devTask 内不变：入口一次性 dcci 读入并合成
        // 期望动态计数，热路径完成判定用缓存值（动态计数走分片），不再反复 dcci 该行。
        // ssbuf 协议下 V 端无判定职责，不缓存；C 端另缓存 AIV 期望动态计数并复位聚合状态
        // （快照须与随后 ssbuf 复位后的零值对齐；tick 复位保证每 devTask 首次空转聚合即判，
        // 空类型预置 flag 及时补发镜像）
        if (IS_AIC) {
            state.drcoCountDynExpect = DrcoGmLoad(&rootFuncList->devTaskCountList.count[DRCO_CORE_TYPE].size) -
                                       DrcoGmLoad(
                                           &rootFuncList->devTaskCountList.count[DRCO_CORE_TYPE].staticExecCount);
            state.warpAggState
                .vectorCountDynExpect = DrcoGmLoad(
                                            &rootFuncList->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIV].size) -
                                        DrcoGmLoad(&rootFuncList->devTaskCountList.count[npu::tile_fwk::DRCO_QUEUE_AIV]
                                                        .staticExecCount);
            state.warpAggState.lastVCount[0] = 0;
            state.warpAggState.lastVCount[1] = 0;
            state.warpAggState.vectorFinishPublished = false;
            state.warpAggState.vectorFinishPollTick = 0;
        }
        state.hasStitchNode = DrcoGmLoad(&rootFuncList->hasStitchNode) != 0;
        __gm__ npu::tile_fwk::PerCorePendingQueue* perCoreQueue = DrcoGmLoad(
            &rootFuncList->perCorePendingQueueArray[BlockDescBlockIdx(state.blockDesc)]);
        DRCO_DCCI_SINGLE_CACHE_LINE(perCoreQueue);

        // ssbuf 段基址（__ssbuf__ 专用地址空间，0 即段起点）：MIX C2V 通信区域，
        // AIC 初始化 [0]/[1] 两个 SPSC 队列与 V 计数/完成镜像区，AIV 按 subblockid 选中消费端
        state.ssbuf = reinterpret_cast<npu::tile_fwk::DrcoSsbufState*>(0);
#if defined(__MIX__)
#if defined(__AIV__)
        state.mixHubC2VReadyQueue = &state.ssbuf->mixHubC2VReadyQueue[get_subblockid() == 1];
        wait_intra_block(PIPE_S, EVENT_ID14);
#else
        npu::tile_fwk::DrcoMixHubC2VReadyQueue::Init(&state.ssbuf->mixHubC2VReadyQueue[0]);
        npu::tile_fwk::DrcoMixHubC2VReadyQueue::Init(&state.ssbuf->mixHubC2VReadyQueue[1]);
        // V 计数/完成镜像复位：须在 set_intra_block 放行 V 之前（跨 devTask 不残留，V 只读不承担复位）
        npu::tile_fwk::DrcoWarpExecuteCount::Init(&state.ssbuf->warpExecuteCount);
        set_intra_block(PIPE_S, EVENT_ID14);                      // 作用于Vec0
        set_intra_block(PIPE_S, EVENT_ID14 + EVENT_NUMS_PER_AIV); // 作用与Vec1
#endif
#endif
        if (DrcoGmLoad(&rootFuncList->totalTaskCount) == 0) {
            uint32_t oldHead = DrcoAtomicLoad(&state.deviceTaskReadyQueue->head);
            DrcoAtomicCasToU32(&state.deviceTaskReadyQueue->head, oldHead, oldHead + 1);
            continue;
        }

        // readymatrix 行游标由全类型 push 核共享的全局游标 readyMatrixPushCol 承担，无需核内 seed

        state.selfReadyQueueSize = 0;
        state.controlFlowCacheBase = rootFuncList->controlFlowCacheBase;
        state.metadataBase = rootFuncList->metadataBase;
        state.programBase = rootFuncList->programBase;
        DrcoMiddleQueueInit(&state, rootFuncList);
        ExecDrcoPerCoreTasks(&state, perCoreQueue, rootFuncList);
        // 头在 perCore 中执行完成（写 0xff 开门）后立即消费 middleQueue——省去进入
        // fetch 自旋到探测到门开的轮询延迟；未开门时 Consume 内部按槽过滤，无害
        DrcoTryMiddleQueueConsume(&state, rootFuncList);
        ExecDrcoReadyQueueTasks(&state, rootFuncList, perCoreQueue);

        SyncAllMix();
        if (BlockDescBlockIdx(state.blockDesc) == 0) {
            DrcoGmStore(&rootFuncList->devTaskFinished, (uint32_t)1);
        }
        DfxProcWhenDevTaskStop(&state.ctx, state.args, state.metric, state.isExectedLeafTask);
    }
    if (BlockDescBlockIdx(state.blockDesc) == 0) {
        state.deviceTaskReadyQueue->head = 0;
        state.deviceTaskReadyQueue->tail = 0;
        uint64_t finished = DrcoGmLoad(&state.runtimeDataRingBufferHeadData->indexFinished.value) + 1;
        DrcoGmStore(&state.runtimeDataRingBufferHeadData->indexFinished.value, finished);
    }

    state.args->taskEntry.reserved[0] = state.ctx.profLevel;
    DfxProcWhenCoreExit(&state.ctx, state.args, state.metric);

    TracePrint(&state);
    return;
}

#endif //__DAV_C310__
#endif
