/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace npu::tile_fwk::Distributed {
class WindowContext {
public:
    WindowContext(int32_t rank, int32_t nrank, void* stream, const std::string& hcclGroupName, const std::string& name,
                  int32_t layer);
    ~WindowContext();
    WindowContext(const WindowContext&) = delete;
    WindowContext& operator=(const WindowContext&) = delete;

    int32_t Rank() const;
    int32_t NRank() const;
    int64_t DataPtr() const;
    // True when all peers share one local UBC_CTP endpoint, so every AIV owns a private
    // jetty and may issue engine writes to any peer. False on multi-endpoint topologies,
    // where each peer has one jetty and a single issuer AIV.
    bool SupportsMultiIssuer() const;
    // True once Connect() has completed. Kernels must not launch before that.
    bool Connected() const;

    // Register one memory region in place (local; before Connect()). protocol is a
    // kPolicyUbMem | kPolicyUrma bitmask; the caller keeps the memory alive.
    void RegisterMemory(int64_t tensorPtr, int64_t size, const std::string& name, int32_t protocol);
    // All-rank collective: register regions, exchange peer bases, build channels.
    void Connect();

    std::string ToString() const;
    void Close(); // all-rank collective teardown; best-effort, never throws

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<WindowContext> CreateWindow(int32_t rank, int32_t nrank, int64_t streamAddr,
                                            const std::string& hcclGroupName, const std::string& name, int32_t layer);

std::string GetGroupTopology(const std::string& hcclGroupName, int32_t rank, int32_t nrank, int32_t layer);
} // namespace npu::tile_fwk::Distributed
