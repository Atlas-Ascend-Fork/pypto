/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_AICPU_INTERFACE_H
#define PYPTO_AICPU_INTERFACE_H

#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include "securec.h"
#include "machine/utils/device_log.h"

/* 该 shim 需要独立加载, 不能依赖 backend server so 中的符号。
 * device 侧不用 DEV_ERROR: 它内部会触发 DEV_ATRACE, 从而引用 DeviceTrace。 */
#ifdef __DEVICE__
#define PYPTO_SHIM_ERROR(errCode, fmt, ...)                                                 \
    do {                                                                                    \
        if (IsLogEnableError()) {                                                           \
            dlog_error(LOG_MOD_ID, "%lu %s\nErrCode: F%05X! " fmt, GET_TID(), __FUNCTION__, \
                       static_cast<uint32_t>(errCode) & 0xFFFFF, ##__VA_ARGS__);            \
        }                                                                                   \
    } while (false)
#else
#define PYPTO_SHIM_ERROR(errCode, fmt, ...) DEV_ERROR(errCode, fmt, ##__VA_ARGS__)
#endif

using TileFwkKernelServelEnty = int (*)(void*);
namespace npu::tile_fwk {
const std::string dynServerKernelFun = "DynTileFwkBackendKernelServer";
const std::string dynServerKernelInitFun = "DynTileFwkBackendKernelServerInit";
const uint64_t dyInitFuncKey = 2;
const uint64_t dyExecFuncKey = 3;
const uint64_t minSoLen = 1;
constexpr uint32_t MemFilePathLen = 64;

class BackendServerHandleManager {
public:
    bool SaveSoFile(char* data, const uint64_t& len, uint8_t deviceId = 0)
    {
        std::lock_guard<std::mutex> lock(funcLock_);
        if (len < minSoLen || firstCreatSo_) {
            DEV_WARN("so len %lu less than minSoLen %lu or first create, no need to copy", len, minSoLen);
            return true;
        }

        std::string memFdName = "libpypto_server" + std::to_string(deviceId) + ".so";
        fileFd_ = memfd_create(memFdName.c_str(), MFD_CLOEXEC);
        DEV_DEBUG("Begin to create server.so in memfd [%s], data ptr [%p], len [%lu]", memFdName.c_str(), data, len);
        if (fileFd_ < 0) {
            int error = errno;
            PYPTO_SHIM_ERROR(DevCommonErr::FILE_ERROR,
                             "#sche.task.pre.init.resource: Create memfd [%s] failed for device[%u], "
                             "errno[%d]: %s",
                             memFdName.c_str(), deviceId, error, std::strerror(error));
            return false;
        }

        // write bin to memfd
        ssize_t writtenSize = write(fileFd_, data, len);
        if (static_cast<uint64_t>(writtenSize) != len) {
            PYPTO_SHIM_ERROR(DevCommonErr::FILE_ERROR,
                             "#sche.task.pre.init.resource: Write memfd [%s] short write, fd[%ld], "
                             "expected[%lu], actual[%ld]",
                             memFdName.c_str(), static_cast<long>(fileFd_), len, static_cast<long>(writtenSize));
            CloseMemFd();
            return false;
        }

        char memFilePath[MemFilePathLen] = {0};
        int ret = snprintf_s(memFilePath, sizeof(memFilePath), sizeof(memFilePath) - 1, "/proc/self/fd/%ld",
                             static_cast<long>(fileFd_));
        if (ret < 0) {
            PYPTO_SHIM_ERROR(DevCommonErr::FILE_ERROR,
                             "#sche.task.pre.init.resource: Build memfd path for [%s] failed, fd[%ld], ret[%d]",
                             memFdName.c_str(), static_cast<long>(fileFd_), ret);
            CloseMemFd();
            return false;
        }
        pyptoServerSoName_ = memFilePath;
        DEV_DEBUG("Create device[%u] server so [%s] success", deviceId, pyptoServerSoName_.c_str());
        firstCreatSo_ = true;
        return true;
    }

    BackendServerHandleManager() = default;

    void SetTileFwkKernelMap()
    {
        std::lock_guard<std::mutex> lock(funcLock_);
        if (firstLoadSo_) {
            return;
        }
        (void)LoadTileFwkKernelFunc(dynServerKernelInitFun);
        (void)LoadTileFwkKernelFunc(dynServerKernelFun);
        firstLoadSo_ = true;
    }

    inline int32_t ExecuteFunc(void* args, const uint64_t funcKey)
    {
        auto func = GetTileFwkKernelFunc(funcKey);
        if (func == nullptr) {
            PYPTO_SHIM_ERROR(DevCommonErr::NULLPTR,
                             "#sche.func.exec: kernel func[%lu] is invalid, cannot get from so %s", funcKey,
                             pyptoServerSoName_.c_str());
            return static_cast<int32_t>(DevCommonErr::NULLPTR);
        }
        return func(args);
    }

    ~BackendServerHandleManager()
    {
        if (soHandle_) {
            DEV_INFO("Close handle [%s]", pyptoServerSoName_.c_str());
            if (dlclose(soHandle_) != 0) {
                const char* error = dlerror();
                DEV_WARN("Close so [%s] failed, dlerror: %s", pyptoServerSoName_.c_str(),
                         error == nullptr ? "unknown" : error);
            }
            soHandle_ = nullptr;
        }
        CloseMemFd();
    }

private:
    void CloseMemFd()
    {
        if (fileFd_ < 0) {
            return;
        }
        if (close(fileFd_) != 0) {
            int error = errno;
            DEV_WARN("Close memfd fd[%ld] failed, errno[%d]: %s", static_cast<long>(fileFd_), error,
                     std::strerror(error));
        }
        fileFd_ = -1;
    }

    void LoadTileFwkKernelFunc(const std::string& kernelName)
    {
        std::string dlOpenError;
        if (soHandle_ == nullptr) {
            soHandle_ = dlopen(pyptoServerSoName_.c_str(), RTLD_LAZY | RTLD_DEEPBIND);
            if (soHandle_ == nullptr) {
                const char* error = dlerror();
                dlOpenError = (error == nullptr) ? "unknown" : error;
            }
            CloseMemFd();
        }
        if (!soHandle_) {
            PYPTO_SHIM_ERROR(DevCommonErr::NULLPTR, "#sche.task.pre.kernel.load: Cannot open so [%s], dlerror: %s",
                             pyptoServerSoName_.empty() ? "<empty path>" : pyptoServerSoName_.c_str(),
                             dlOpenError.c_str());
            return;
        }
        uint64_t funcKey = 0;
        if (kernelName == dynServerKernelInitFun) {
            funcKey = dyInitFuncKey;
        } else if (kernelName == dynServerKernelFun) {
            funcKey = dyExecFuncKey;
        }
        DEV_DEBUG("Current to open kernel func: name=%s, funcKey=%lu.", kernelName.c_str(), funcKey);
        auto iter = kernelKey2FuncHandle_.find(funcKey);
        if (iter != kernelKey2FuncHandle_.end()) {
            return;
        }

        TileFwkKernelServelEnty tileFwkServrFuncEnty = reinterpret_cast<TileFwkKernelServelEnty>(
            dlsym(soHandle_, kernelName.c_str()));
        if (tileFwkServrFuncEnty == nullptr) {
            const char* error = dlerror();
            PYPTO_SHIM_ERROR(DevCommonErr::NULLPTR,
                             "#sche.task.pre.kernel.load: Cannot find symbol [%s] in so [%s], dlerror: %s",
                             kernelName.c_str(), pyptoServerSoName_.c_str(), (error == nullptr) ? "unknown" : error);
            if (dlclose(soHandle_) != 0) {
                const char* closeError = dlerror();
                DEV_WARN("Close so [%s] failed, dlerror: %s", pyptoServerSoName_.c_str(),
                         closeError == nullptr ? "unknown" : closeError);
            }
            soHandle_ = nullptr;
            return;
        }
        DEV_INFO("kernelName=%s has been loaded", kernelName.c_str());
        kernelKey2FuncHandle_[funcKey] = tileFwkServrFuncEnty;
        return;
    }

    TileFwkKernelServelEnty GetTileFwkKernelFunc(const uint64_t funcKey)
    {
        auto iter = kernelKey2FuncHandle_.find(funcKey);
        if (iter != kernelKey2FuncHandle_.end()) {
            return iter->second;
        }
        PYPTO_SHIM_ERROR(DevCommonErr::NULLPTR, "#sche.kernel.check: Function[%lu] is null.", funcKey);
        return nullptr;
    }

    std::unordered_map<uint64_t, TileFwkKernelServelEnty> kernelKey2FuncHandle_;
    std::mutex funcLock_;
    void* soHandle_ = nullptr;
    bool firstCreatSo_ = false;
    bool firstLoadSo_ = false;
    int64_t fileFd_ = -1;
    std::string pyptoServerSoName_;
};

} // namespace npu::tile_fwk
extern "C" {
__attribute__((visibility("default"))) uint32_t DynPyptoKernelServer(void* args);
__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerInit(void* args);
}
#endif // TILE_FWK_AICPU_INTERFACE_H
