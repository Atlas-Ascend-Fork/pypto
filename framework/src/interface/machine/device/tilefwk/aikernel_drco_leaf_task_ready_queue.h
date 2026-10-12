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

// ssbuf 段（__ssbuf__ 专用地址空间，段基址 0 合法）上的 C2V 通信状态：
// 每个 AIC 块两个 SPSC 队列（配对 V0/V1 各消费一个）
struct DrcoSsbufState {
    DrcoMixHubC2VReadyQueue mixHubC2VReadyQueue[2];
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

} // namespace npu::tile_fwk

#endif
