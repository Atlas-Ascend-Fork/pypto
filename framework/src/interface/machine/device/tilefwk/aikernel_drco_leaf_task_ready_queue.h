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
 * \file aikernel_drco_leaf_task_ready_queue.h
 * \brief
 */

#ifndef AIKERNEL_DRCO_LEAF_TASK_READY_QUEUE_H
#define AIKERNEL_DRCO_LEAF_TASK_READY_QUEUE_H

#include "tilefwk/aikernel_tensor.h"

namespace npu::tile_fwk {

/*
 * DRCO = Dependency Resolving by aiCOre
 * DRCU = Dependency Resolving by aiCpU
 */

using LeafTaskId = uint32_t;

struct PerCoreReadyList {
    uint32_t size;
    LeafTaskId taskList[0];
};

// SPSC = Single Producer Single Consumer
template <uint64_t bufSize = 6>
struct DrcoSsbufSPSCQueue {
    static constexpr uint32_t RING_BUF_SIZE = bufSize;
    static constexpr uint32_t ELEM_OFFSET = 2;
    /* [0] = head, 只有 producer 写
     * [1] = tail, 只有 consumer 读
     */
    uint32_t elems[2 + RING_BUF_SIZE];

#ifndef __TILE_FWK_HOST__
    static inline void Init(DrcoSsbufSPSCQueue* addr)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        ptr[0] = 0;
        ptr[1] = 0;
        __asm__ volatile("DSB #0");
#else
        (void)addr;
#endif
    }

    // C(AIC)调用：写入一个元素，满则返回false
    static inline bool Push(DrcoSsbufSPSCQueue* addr, uint32_t elem)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curHead = ptr[0];
        uint32_t curTail = ptr[1];
        if (curTail + RING_BUF_SIZE <= curHead) {
            return false; // 满
        }
        ptr[ELEM_OFFSET + (curHead % RING_BUF_SIZE)] = elem;
        __asm__ volatile("DSB #0");
        ptr[0] = curHead + 1;
        __asm__ volatile("DSB #0");
        return true;
#else
        (void)addr;
        (void)elem;
        return false;
#endif
    }

    // V(AIV)调用：读出一个元素，空则返回false
    static inline bool Pop(DrcoSsbufSPSCQueue* addr, uint32_t& elem)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curTail = ptr[1];
        uint32_t curHead = ptr[0];
        if (curTail == curHead) {
            return false; // 空
        }
        elem = ptr[ELEM_OFFSET + (curTail % RING_BUF_SIZE)];
        ptr[1] = curTail + 1;
        __asm__ volatile("DSB #0");
        return true;
#else
        (void)addr;
        (void)elem;
        return false;
#endif
    }

    static inline uint32_t Size(DrcoSsbufSPSCQueue* addr)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curHead = ptr[0];
        uint32_t curTail = ptr[1];
        return (curHead + RING_BUF_SIZE - curTail) % RING_BUF_SIZE;
#else
        (void)addr;
        return 0;
#endif
    }
#endif
};

/* per core pending queue mechanism can be used in both codr and cudr */
using DrcoMixHubC2VReadyQueue = DrcoSsbufSPSCQueue<6>;

// 1C2V 组内 V→C 计数聚合与 C→V 完成镜像（ssbuf 段，仅 3510 有读写语义）：
// vCount[2] = V0/V1 各自单写者计数（同组 C 读差值入 GM AIV 分片）；finishState = C 发布的
// 完成镜像位域（FINISH_VECTOR = AIV 类型完成，V 跳过本类型 leaf 消费；FINISH_ALL = 双类型
// 完成，V 全局退出）。写序 值 → DSB，读侧 volatile 防跨轮提升；复位由同组 C 在 devTask
// 入口、EVENT_ID14 放行 V 之前执行（跨 devTask 不残留）
struct DrcoWarpExecuteCount {
    // finishState 位定义（C 单写者置位幂等；V 只读）
    static constexpr uint32_t FINISH_VECTOR = 0x1; // AIV(vector) 类型完成镜像（V own-done 门控）
    static constexpr uint32_t FINISH_ALL = 0x2;    // 双类型完成镜像（V 全局退出）

    uint32_t vCount[2];
    uint32_t finishState;
#ifndef __TILE_FWK_HOST__
    static inline void Init(DrcoWarpExecuteCount* addr)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        ptr[0] = 0;
        ptr[1] = 0;
        ptr[2] = 0;
        __asm__ volatile("DSB #0");
#else
        (void)addr;
#endif
    }

    // V 端调用：本核分片计数自增（单写者，普通读改写即可，无需 GM 原子）
    static inline void VCountAdd(DrcoWarpExecuteCount* addr, uint32_t idx, uint32_t count)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (count == 0) {
            return;
        }
        __ssbuf__ volatile uint32_t* ptr = (__ssbuf__ volatile uint32_t*)addr;
        ptr[idx] += count;
        __asm__ volatile("DSB #0");
#else
        (void)addr;
        (void)idx;
        (void)count;
#endif
    }

    // C 端调用：读组内 V 计数
    static inline uint32_t VCountLoad(DrcoWarpExecuteCount* addr, uint32_t idx)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ volatile uint32_t* ptr = (__ssbuf__ volatile uint32_t*)addr;
        return ptr[idx];
#else
        (void)addr;
        (void)idx;
        return 0;
#endif
    }

    // C 端调用：完成镜像位置位（幂等；bits 取 FINISH_VECTOR / FINISH_ALL，单调置位不清）
    static inline void FinishPublish(DrcoWarpExecuteCount* addr, uint32_t bits)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ volatile uint32_t* ptr = (__ssbuf__ volatile uint32_t*)addr;
        ptr[2] |= bits;
        __asm__ volatile("DSB #0");
#else
        (void)addr;
        (void)bits;
#endif
    }

    // V 端调用：读完成镜像字（一轮一次，FINISH_VECTOR / FINISH_ALL 位复用）
    static inline uint32_t FinishStateLoad(DrcoWarpExecuteCount* addr)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ volatile uint32_t* ptr = (__ssbuf__ volatile uint32_t*)addr;
        return ptr[2];
#else
        (void)addr;
        return 0;
#endif
    }
#endif
};

// 并行窗口槽位数：与 SCH_DEVTASK_MAX_PARALLELISM 数值对齐（aikernel_data.h 有 static_assert 强校验。
constexpr uint32_t DRCO_PARALLEL_WINDOW_SLOT_NUM = 8;

// ssbuf 段（段基址 0 合法）上的 C2V 通信状态：每窗口槽一对 SPSC 队列（配对 V0/V1 各消费一个），8 槽 × 2 环 × 32B =
// 512B/核；
struct DrcoSsbufState {
    DrcoMixHubC2VReadyQueue mixHubC2VReadyQueue[DRCO_PARALLEL_WINDOW_SLOT_NUM][2];
    DrcoWarpExecuteCount warpExecuteCount;
};

/* per core pending queue mechanism can be used in both codr and cudr */
struct PerCorePendingQueue {
    uint32_t head;
    uint32_t tail;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    PerCorePendingQueue() : head(0), tail(0), size(0) {}

    void UnsafeEnqueue(LeafTaskId task) { taskList[size++] = task; }
#endif
};

struct DrcoLocalReadyQueue {
    uint32_t head;
    uint8_t pad[64 - sizeof(uint32_t)];
    uint32_t tail;
    uint8_t pad2[64 - sizeof(uint32_t)];
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    explicit DrcoLocalReadyQueue(uint32_t capacity) : head(0), tail(0), size(capacity) {}
#endif
};

// 队列分组大小 = 组内核数（localReadyQueueArray 按 4 核一组分池；LocalMatrix 已合并为
// 每类型单实例，不再使用分组）
constexpr uint32_t LOCAL_GROUP_SIZE = 4;

// 合并版 LocalMatrix：每类型单实例（AIC/AIV/MIX 各一），行 = 消费者（类型内本地编号，
// AIC/MIX: blockIdx ∈ [0, nrValidAic)，AIV: blockIdx - nrValidAic）：push 逐槽 u32 CAS
// 写入（0→任务），pop 先 dcci 失效读探测（空字跳过整字原子，省空转 RMW 写流量）、
// 非零字 u64 exch 整字读清（一次原子取相邻 2 任务）；行私有 64B 布局下逐字 dcci 保证
// L1 可见性，亦无"提示字与数据分离"的滞留窗口。
// 行内 taskList[M] 槽连续（列 = 缓冲深度）：每消费者可囤 colNum 个就绪任务，整矩阵
// 容量 = colNum × validCoreNum，超出溢出到 local ready queue；colNum 恒为偶数
// （pop 整字 exch 需相邻两槽 8B 对齐），host 按解依赖扇出统计在
// [DRCO_MATRIX_COL_DEFAULT, DRCO_MATRIX_COL_MAX] 内选定（≤ ColSize）。
// slot 语义：0 = 空闲，非 0 为 DRCO_ENCODE_TASK 编码的任务
constexpr uint32_t DRCO_MATRIX_COL_DEFAULT = 4;
constexpr uint32_t DRCO_MATRIX_COL_MAX = 8;
static_assert(DRCO_MATRIX_COL_DEFAULT % 2 == 0 && DRCO_MATRIX_COL_MAX % 2 == 0,
              "matrix colNum must be even for u64 word exch alignment");

constexpr uint32_t CACHE_LINE_SIZE = 64;

// 就绪矩阵通用布局：行 = 消费者（行数上限 N），每行 taskList[M] 槽为该消费者的缓冲深度，
// 行按 cacheline 对齐 + 尾部填充独占整数个 64B——消费者行间的原子 RMW 不共享 cacheline；
// 行首 64B 对齐保证行内槽对 8B 对齐（pop 整字 u64 exch 前提；slab 池分配恒 64B 对齐）；
// 实际列数 colNum（每行缓冲深度）/ 有效行数 validCoreNum 由 host 分配时写入尾部
// （validCoreNum = 0 表示本类型无核，禁止写入）
template <typename T, unsigned N, unsigned M>
struct DrcoGlobalReadyMatrixBase {
    enum { RowSize = N, ColSize = M };
    typedef T ElementType;

    struct alignas(CACHE_LINE_SIZE) Row {
        T taskList[M];
        uint8_t pad[(sizeof(T) * M + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE - sizeof(T) * M];
    } rowList[N];

    uint32_t colNum;
    uint32_t validCoreNum;

#ifdef __TILE_FWK_HOST__
    DrcoGlobalReadyMatrixBase(uint32_t colCount, uint32_t validCoreCount)
        : colNum(colCount), validCoreNum(validCoreCount)
    {
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < M; j++) {
                rowList[i].taskList[j] = 0;
            }
        }
    }
#endif
};

#define DRCO_ENCODE_TASK(task) ((task) + 1)
#define DRCO_DECODE_TASK(task) ((task) - 1)

struct PerCoreMiddleQueue {
    uint32_t headTaskId;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    PerCoreMiddleQueue() : headTaskId(0), size(0) {}
    void UnsafeEnqueue(LeafTaskId task) { taskList[size++] = task; }
#endif
};

} // namespace npu::tile_fwk

#endif
