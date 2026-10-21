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
 * \file fed_prof_holder.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "interface/machine/device/tilefwk/aikernel_define.h"
#include "interface/machine/device/tilefwk/aikernel_profile.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct DeviceArgs;

namespace npu::tile_fwk {

enum class ArchInfo : uint32_t;

struct FedProfDeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint32_t scheCpuNum{0};
    uint64_t deviceId{0};
    uint64_t sharedBuffer{0};
    uint64_t devDfxArgAddr{0};
    ArchInfo archInfo{};

    FedProfDeviceArgs() = default;
    explicit FedProfDeviceArgs(const DeviceArgs& args);
    uint64_t GetBlockNum() const { return nrValidAic * (nrAiv / nrAic + 1); }
};

struct FedProfLaunchArgs {
    uint32_t launchRoundBegin;
    uint32_t launchRoundEnd;
};

class FedProfHolderBase {
public:
    FedProfHolderBase() = default;
    virtual ~FedProfHolderBase() = default;

    virtual uint64_t DeviceDataAllocate(const FedProfDeviceArgs& devArgs)
    {
        UNUSED(devArgs);
        return 0;
    }
    virtual void DeviceDataClear();

    template <typename T>
    void DumpJson(json& perCoreProfObjList, const std::vector<uint8_t>& hostBuffer, const std::vector<T*>& dataList,
                  const FedProfDeviceArgs& devArgs, const FedProfLaunchArgs& launchArgs)
    {
        std::vector<FedProfDataBase*> baseList(dataList.begin(), dataList.end());
        DumpJsonBase(perCoreProfObjList, hostBuffer, baseList, devArgs, launchArgs);
    }

    template <typename T>
    void DeviceDataGet(std::vector<uint8_t>& hostBuffer, std::vector<T*>& dataList, const FedProfDeviceArgs& devArgs,
                       FedProfLaunchArgs launchArgs)
    {
        std::vector<FedProfDataBase*> baseList;
        DeviceDataGetBase(hostBuffer, baseList, devArgs, launchArgs);
        dataList.resize(baseList.size());
        for (size_t index = 0; index < baseList.size(); ++index) {
            dataList[index] = static_cast<T*>(baseList[index]);
        }
    }
    void DeviceDataRelease();

    uint64_t GetDevAddr() const { return devAddr_; }
    bool IsAllocated() const { return allocated_; }

protected:
    virtual void DumpJsonBase(json& perCoreProfObjList, const std::vector<uint8_t>& hostBuffer,
                              const std::vector<FedProfDataBase*>& dataList, const FedProfDeviceArgs& devArgs,
                              const FedProfLaunchArgs& launchArgs)
    {
        UNUSED(perCoreProfObjList, hostBuffer, dataList, devArgs, launchArgs);
    }
    virtual void DeviceDataGetBase(std::vector<uint8_t>& hostBuffer, std::vector<FedProfDataBase*>& dataList,
                                   const FedProfDeviceArgs& devArgs, FedProfLaunchArgs launchArgs)
    {
        UNUSED(hostBuffer, dataList, devArgs, launchArgs);
    }
    uint64_t devAddr_{0};
    bool allocated_{false};
};

template <FedProfKind _kind>
class FedProfHolder : public FedProfHolderBase {
public:
    enum {
        kind = _kind,
    };
};

class FedProfPerCoreProfHolder : public FedProfHolder<FedProfKind::FED_PROF_PER_CORE_PROF> {
public:
    virtual uint64_t DeviceDataAllocate(const FedProfDeviceArgs&) override;
    virtual void DeviceDataClear() override;
    void DumpJsonAicpuProf(json& launchObjList, const std::vector<uint8_t>& hostBuffer,
                           const std::vector<FedProfDataBase*>& dataList, const FedProfDeviceArgs& devArgs,
                           const FedProfLaunchArgs& launchArgs);

protected:
    virtual void DumpJsonBase(json& perCoreProfObjList, const std::vector<uint8_t>& hostBuffer,
                              const std::vector<FedProfDataBase*>& dataList, const FedProfDeviceArgs& devArgs,
                              const FedProfLaunchArgs& launchArgs) override;
    virtual void DeviceDataGetBase(std::vector<uint8_t>& hostBuffer, std::vector<FedProfDataBase*>& dataList,
                                   const FedProfDeviceArgs& devArgs, FedProfLaunchArgs launchArgs) override;
    uint64_t dataCount_{0};
};

class FedProfAicpuProfHolder : public FedProfHolder<FedProfKind::FED_PROF_AICPU_PROF> {
public:
    virtual uint64_t DeviceDataAllocate(const FedProfDeviceArgs&) override;
    virtual void DeviceDataClear() override;

protected:
    virtual void DumpJsonBase(json& launchObjList, const std::vector<uint8_t>& hostBuffer,
                              const std::vector<FedProfDataBase*>& dataList, const FedProfDeviceArgs& devArgs,
                              const FedProfLaunchArgs& launchArgs) override;
    virtual void DeviceDataGetBase(std::vector<uint8_t>& hostBuffer, std::vector<FedProfDataBase*>& dataList,
                                   const FedProfDeviceArgs& devArgs, FedProfLaunchArgs launchArgs) override;
};

} // namespace npu::tile_fwk
