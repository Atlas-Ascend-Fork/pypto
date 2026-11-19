/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include "core/logging.h"
#include "pypto_pro/backend/common/cann_runtime.h"

using pypto::backend::KernelParamKind;
static constexpr int32_t MAX_KERNEL_BUF_LEN = 1024;
static constexpr int32_t MAX_TENSOR_NUM = 128;
static constexpr int32_t MAX_SCHEMA_DIMS = 8;
static constexpr uint32_t MAX_RUNTIME_ARGS_SIZE = 1024U * 1024U;

struct ExceptionLaunchInfo {
    std::string kernelName;
    void* argAddr = nullptr;
    uint32_t argsize = 0U;
    void* bin = nullptr;
};

// ACL exposes exception arguments through accessors, so this library does not
// need to depend on the adapter's private RtExceptionInfo layout.
struct AclExceptionApi {
    using GetArgs = int32_t (*)(const void*, void**, uint32_t*);
    using GetFuncHandle = int32_t (*)(const void*, void**);
    using GetFunctionName = int32_t (*)(void*, uint32_t, char*);
    using FunctionGetBinary = int32_t (*)(void*, void**);
    using Memcpy = int32_t (*)(void*, size_t, const void*, size_t, int32_t);

    GetArgs getArgs = nullptr;
    GetFuncHandle getFuncHandle = nullptr;
    GetFunctionName getFunctionName = nullptr;
    FunctionGetBinary functionGetBinary = nullptr;
    Memcpy memcpy = nullptr;
};

static const AclExceptionApi& GetAclExceptionApi()
{
    static const AclExceptionApi api = []() {
        AclExceptionApi result;
        void* handle = dlopen("libacl_rt.so", RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            return result;
        }
        result.getArgs = reinterpret_cast<AclExceptionApi::GetArgs>(dlsym(handle, "aclrtGetArgsFromExceptionInfo"));
        result.getFuncHandle = reinterpret_cast<AclExceptionApi::GetFuncHandle>(
            dlsym(handle, "aclrtGetFuncHandleFromExceptionInfo"));
        result.getFunctionName = reinterpret_cast<AclExceptionApi::GetFunctionName>(
            dlsym(handle, "aclrtGetFunctionName"));
        result.functionGetBinary = reinterpret_cast<AclExceptionApi::FunctionGetBinary>(
            dlsym(handle, "aclrtFunctionGetBinary"));
        result.memcpy = reinterpret_cast<AclExceptionApi::Memcpy>(dlsym(RTLD_DEFAULT, "aclrtMemcpy"));
        if (result.memcpy == nullptr) {
            result.memcpy = reinterpret_cast<AclExceptionApi::Memcpy>(dlsym(handle, "aclrtMemcpy"));
        }
        return result;
    }();
    return api;
}

static bool GetKernelExceptionArgs(const void* exceptionInfo, ExceptionLaunchInfo& info)
{
    const auto& api = GetAclExceptionApi();
    if (exceptionInfo == nullptr || api.getArgs == nullptr || api.getFuncHandle == nullptr ||
        api.getFunctionName == nullptr || api.functionGetBinary == nullptr) {
        return false;
    }
    void* funcHandle = nullptr;
    char kernelName[MAX_KERNEL_BUF_LEN] = {};
    if (api.getArgs(exceptionInfo, &info.argAddr, &info.argsize) != 0 ||
        api.getFuncHandle(exceptionInfo, &funcHandle) != 0 || funcHandle == nullptr ||
        api.getFunctionName(funcHandle, MAX_KERNEL_BUF_LEN - 1U, kernelName) != 0 || kernelName[0] == '\0' ||
        api.functionGetBinary(funcHandle, &info.bin) != 0) {
        return false;
    }
    info.kernelName = kernelName;
    return true;
}

enum class AdxTensorType : int32_t { INPUT, OUTPUT, WORKSPACE };
enum class AdxAddressType : int32_t { TRADITIONAL, NOTILING, RAW };
enum class AdxExceptionDumpMode : uint32_t {
    ADX_DUMP_MODE_NONE = 0,
    ADX_DUMP_MODE_OVERWRITE = 1,
    ADX_DUMP_MODE_ADDITIONAL = 2,
};

struct AdxTensorInfo {
    AdxTensorType type;
    size_t tensorSize;
    int32_t format;
    int32_t dataType;
    int64_t* tensorAddr;
    AdxAddressType addrType;
    int32_t placement;
    uint32_t argsOffSet;
    std::vector<int64_t> shape;
    std::vector<int64_t> originShape;
};

struct AdxExceptionDumpInfo {
    uint32_t coreId;
    int32_t coreType;
    uint32_t argssize;
    void* argAddr;
    void* bin;
    char kernelName[MAX_KERNEL_BUF_LEN];
    char kernelDisplayName[MAX_KERNEL_BUF_LEN];
    uint32_t extraTensorNum;
    AdxTensorInfo tensorInfo[MAX_TENSOR_NUM];
};

struct DumpParamSchema {
    std::string name;
    std::string dtypeName;
    KernelParamKind kind;
    AdxTensorType tensorType;
    int32_t dataType;
    int32_t elementBits;
    uint32_t argSlot;
    size_t runtimeSize = 0U;
    std::vector<int64_t> shape;
    std::vector<int32_t> dynamicSlots;
};

struct DumpKernelSchema {
    std::vector<DumpParamSchema> params;
    std::string debugCmd;
    bool debugCmdExecuted = false;
    uint32_t blockDim = 1U;
};

using AdumpExceptionDumpCallback = int32_t (*)(void*, AdxExceptionDumpInfo*, uint32_t, uint32_t*,
                                               AdxExceptionDumpMode*);
using AdumpRegExceptionDumpCallBack = int32_t (*)(AdumpExceptionDumpCallback);

static bool g_registered = false;
static void* g_adumpHandle = nullptr;
static AdumpRegExceptionDumpCallBack g_regFunc = nullptr;
static std::unordered_map<std::string, DumpKernelSchema> g_dumpSchemas;
static std::mutex g_dumpSchemasMutex;

static std::string GetSchemaKernelName(const std::string& runtimeName)
{
    // JIT registers the IR function name, while Runtime reports the Itanium-mangled
    // global kernel symbol (and may append a mixed-core suffix).
    if (runtimeName.compare(0, 2, "_Z") != 0) {
        return runtimeName;
    }
    size_t pos = 2;
    size_t nameLength = 0;
    while (pos < runtimeName.size() && runtimeName[pos] >= '0' && runtimeName[pos] <= '9') {
        const size_t digit = static_cast<size_t>(runtimeName[pos] - '0');
        if (nameLength > (std::numeric_limits<size_t>::max() - digit) / 10) {
            return runtimeName;
        }
        nameLength = nameLength * 10 + digit;
        ++pos;
    }
    return nameLength > 0 && nameLength <= runtimeName.size() - pos ? runtimeName.substr(pos, nameLength) : runtimeName;
}

static AdumpRegExceptionDumpCallBack LoadAdumpRegFunc()
{
    if (g_regFunc != nullptr) {
        return g_regFunc;
    }
    const char* cannPath = std::getenv("ASCEND_CANN_PACKAGE_PATH");
    if (cannPath == nullptr) {
        cannPath = std::getenv("ASCEND_HOME_PATH");
    }
    if (cannPath != nullptr) {
        std::string soPath = std::string(cannPath) + "/lib64/libascend_dump.so";
        g_adumpHandle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    if (g_adumpHandle == nullptr) {
        g_adumpHandle = dlopen("libascend_dump.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (g_adumpHandle == nullptr) {
        return nullptr;
    }
    g_regFunc = reinterpret_cast<AdumpRegExceptionDumpCallBack>(
        dlsym(g_adumpHandle,
              "_ZN3Adx29AdumpRegExceptionDumpCallbackEPFjPvPNS_17ExceptionDumpInfoEjPjPNS_17ExceptionDumpModeEE"));
    return g_regFunc;
}

static void WriteLaunchMetaFile(const char* kernelName, const char* launchMeta)
{
    if (kernelName == nullptr || kernelName[0] == '\0' || launchMeta == nullptr || launchMeta[0] == '\0') {
        return;
    }
    // ASCEND_WORK_PATH may be unset; fall back to the process cwd so the
    // launch meta lands next to CANN's own dump data files (which use the
    // same default).
    char cwdBuf[4096] = {0};
    const char* workPath = std::getenv("ASCEND_WORK_PATH");
    if (workPath == nullptr || workPath[0] == '\0') {
        if (getcwd(cwdBuf, sizeof(cwdBuf)) == nullptr) {
            IR_LOGW() << "ProExceptionDumpCallback: getcwd failed, launch meta skipped";
            return;
        }
        workPath = cwdBuf;
    }
    const char* deviceId = std::getenv("TILE_FWK_DEVICE_ID");
    std::string devId = (deviceId != nullptr && deviceId[0] != '\0') ? deviceId : "0";
    std::string dumpDir = std::string(workPath) + "/extra-info/data-dump/" + devId;
    std::string metaPath = dumpDir + "/" + kernelName + "_launch_args.json";
    FILE* file = std::fopen(metaPath.c_str(), "w");
    if (file == nullptr) {
        IR_LOGW() << "ProExceptionDumpCallback: write launch meta failed: " << metaPath;
        return;
    }
    std::fwrite(launchMeta, 1, std::strlen(launchMeta), file);
    std::fclose(file);
}

static bool ResolveSchemaTensor(const DumpParamSchema& param, const uint64_t* args, uint32_t argSlots,
                                AdxTensorInfo& info)
{
    if (param.kind == KernelParamKind::WINDOW) {
        return false; // The descriptor size and contents are owned by distributed runtime.
    }
    if (param.argSlot >= argSlots || param.shape.size() != param.dynamicSlots.size()) {
        return false;
    }
    const uint64_t address = args[param.argSlot];
    if (address == 0U) {
        return false;
    }

    std::vector<int64_t> shape;
    uint64_t tensorBytes = param.runtimeSize;
    if (param.kind == KernelParamKind::TENSOR) {
        uint64_t numel = 1U;
        shape.reserve(param.shape.size());
        for (size_t i = 0; i < param.shape.size(); ++i) {
            int64_t extent = param.shape[i];
            const int32_t dynamicSlot = param.dynamicSlots[i];
            if (dynamicSlot >= 0) {
                if (static_cast<uint32_t>(dynamicSlot) >= argSlots) {
                    return false;
                }
                extent = static_cast<int64_t>(args[dynamicSlot]);
            }
            if (extent <= 0 || numel > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(extent)) {
                return false;
            }
            numel *= static_cast<uint64_t>(extent);
            shape.push_back(extent);
        }

        const uint64_t elementBits = static_cast<uint64_t>(param.elementBits);
        if (elementBits == 0U || numel > (std::numeric_limits<uint64_t>::max() - 7U) / elementBits) {
            return false;
        }
        tensorBytes = (numel * elementBits + 7U) / 8U;
    } else if (tensorBytes == 0U) {
        return false;
    }
    const size_t tensorSize = static_cast<size_t>(tensorBytes);
    if (static_cast<uint64_t>(tensorSize) != tensorBytes) {
        return false;
    }

    info.type = param.tensorType;
    info.tensorSize = tensorSize;
    info.format = 0;
    info.dataType = param.dataType;
    info.tensorAddr = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(address));
    info.addrType = AdxAddressType::TRADITIONAL;
    info.placement = 0;
    info.argsOffSet = 0U;
    info.shape = shape;
    info.originShape = std::move(shape);
    return true;
}

static void AppendScalarJson(std::ostringstream& stream, int32_t dataType, uint64_t value)
{
    switch (dataType) {
        case 0: { // ACL_FLOAT
            uint32_t bits = static_cast<uint32_t>(value);
            float scalar = 0.0F;
            std::memcpy(&scalar, &bits, sizeof(scalar));
            if (std::isnan(scalar)) {
                stream << "NaN";
            } else if (std::isinf(scalar)) {
                stream << (scalar < 0.0F ? "-Infinity" : "Infinity");
            } else {
                // Keep -0.0 as a floating-point JSON value; Python parses -0 as integer zero.
                stream << std::showpoint << std::setprecision(std::numeric_limits<float>::max_digits10) << scalar
                       << std::noshowpoint;
            }
            break;
        }
        case 2: // ACL_INT8
        case 3: // ACL_INT32
        case 6: // ACL_INT16
        case 9: // ACL_INT64
            stream << static_cast<int64_t>(value);
            break;
        case 12: // ACL_BOOL
            stream << (value == 0U ? "false" : "true");
            break;
        default:
            stream << value;
            break;
    }
}

static std::string BuildSchemaLaunchMeta(const char* kernelName, const DumpKernelSchema& schema, const uint64_t* args,
                                         uint32_t argSlots)
{
    std::ostringstream stream;
    stream << "{\"kernel_name\":\"" << kernelName << "\",\"block_dim\":" << schema.blockDim << ",\"abi\":[";
    bool first = true;
    for (const auto& param : schema.params) {
        if (param.argSlot >= argSlots) {
            return "";
        }
        if (!first) {
            stream << ',';
        }
        first = false;
        if (param.kind == KernelParamKind::WINDOW) {
            stream << "{\"name\":\"" << param.name << "\",\"kind\":\"window\",\"addr\":" << args[param.argSlot] << '}';
        } else if (param.kind != KernelParamKind::SCALAR) {
            AdxTensorInfo info;
            if (!ResolveSchemaTensor(param, args, argSlots, info)) {
                stream << "{\"name\":\"" << param.name << "\",\"kind\":\"ptr\",\"dtype\":\"" << param.dtypeName
                       << "\",\"addr\":0,\"size\":0,\"shape\":[]}";
                continue;
            }
            stream << "{\"name\":\"" << param.name << "\",\"kind\":\"ptr\",\"dtype\":\"" << param.dtypeName
                   << "\",\"addr\":" << reinterpret_cast<uintptr_t>(info.tensorAddr) << ",\"size\":" << info.tensorSize
                   << ",\"shape\":[";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                if (i != 0U) {
                    stream << ',';
                }
                stream << info.shape[i];
            }
            stream << "]}";
        } else {
            stream << "{\"name\":\"" << param.name << "\",\"kind\":\"scalar\",\"value\":";
            AppendScalarJson(stream, param.dataType, args[param.argSlot]);
            stream << '}';
        }
    }
    stream << "]}";
    return stream.str();
}

static bool ValidateRuntimeArgs(const ExceptionLaunchInfo& runtimeArgs)
{
    const char* runtimeKernelName = runtimeArgs.kernelName.c_str();
    if (runtimeKernelName[0] == '\0') {
        IR_LOGW() << "ProExceptionDumpCallback: runtime kernel name unavailable; dump skipped";
        return false;
    }
    const std::string kernelName = GetSchemaKernelName(runtimeKernelName);
    const char* invalidArgsReason = nullptr;
    if (runtimeArgs.argAddr == nullptr) {
        invalidArgsReason = "null launch argument address";
    } else if (runtimeArgs.argsize == 0U) {
        invalidArgsReason = "empty launch argument buffer";
    } else if (runtimeArgs.argsize > MAX_RUNTIME_ARGS_SIZE) {
        invalidArgsReason = "launch argument buffer exceeds size limit";
    } else if (runtimeArgs.argsize % sizeof(uint64_t) != 0U) {
        invalidArgsReason = "launch argument size is not 8-byte aligned";
    }
    if (invalidArgsReason != nullptr) {
        IR_LOGW() << "ProExceptionDumpCallback: dump skipped for runtime kernel '" << runtimeKernelName
                  << "' (schema name '" << kernelName << "'): " << invalidArgsReason
                  << " (argsize=" << runtimeArgs.argsize << ")";
        return false;
    }
    return true;
}

static int32_t ProExceptionDumpCallbackImpl(const ExceptionLaunchInfo& runtimeArgs, const uint64_t* runtimeArgSlots,
                                            uint32_t runtimeArgSlotCount, AdxExceptionDumpInfo* exceptionDumpInfo,
                                            uint32_t exceptionDumpSize, uint32_t* exceptionDumpRealSize,
                                            AdxExceptionDumpMode* mode)
{
    *mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    *exceptionDumpRealSize = 0U;
    const char* runtimeKernelName = runtimeArgs.kernelName.c_str();
    const std::string kernelName = GetSchemaKernelName(runtimeKernelName);

    DumpKernelSchema schema;
    bool runDebugCmd = false;
    {
        std::lock_guard<std::mutex> lock(g_dumpSchemasMutex);
        auto it = g_dumpSchemas.find(kernelName);
        if (it == g_dumpSchemas.end()) {
            IR_LOGW() << "ProExceptionDumpCallback: no registered schema for runtime kernel '" << runtimeKernelName
                      << "' (normalized name '" << kernelName << "'); PyPTO dump skipped";
            return 0;
        }
        schema = it->second;
        if (!it->second.debugCmd.empty() && !it->second.debugCmdExecuted) {
            runDebugCmd = true;
            it->second.debugCmdExecuted = true;
        }
    }

    *mode = AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE;

    for (uint32_t i = 0U; i < exceptionDumpSize; i++) {
        exceptionDumpInfo[i].coreId = 0U;
        exceptionDumpInfo[i].coreType = 0;
        exceptionDumpInfo[i].argAddr = runtimeArgs.argAddr;
        exceptionDumpInfo[i].argssize = runtimeArgs.argsize;
        exceptionDumpInfo[i].bin = runtimeArgs.bin;
        snprintf(exceptionDumpInfo[i].kernelName, MAX_KERNEL_BUF_LEN, "%s", kernelName.c_str());
        snprintf(exceptionDumpInfo[i].kernelDisplayName, MAX_KERNEL_BUF_LEN, "%s", kernelName.c_str());

        uint32_t idx = 0U;
        for (const auto& param : schema.params) {
            if (idx >= MAX_TENSOR_NUM) {
                break;
            }
            if (param.kind != KernelParamKind::SCALAR && param.kind != KernelParamKind::WINDOW &&
                ResolveSchemaTensor(param, runtimeArgSlots, runtimeArgSlotCount,
                                    exceptionDumpInfo[i].tensorInfo[idx])) {
                idx++;
            }
        }
        exceptionDumpInfo[i].extraTensorNum = idx;
    }

    *exceptionDumpRealSize = exceptionDumpSize;

    const std::string launchMeta = BuildSchemaLaunchMeta(kernelName.c_str(), schema, runtimeArgSlots,
                                                         runtimeArgSlotCount);
    WriteLaunchMetaFile(kernelName.c_str(), launchMeta.c_str());

    if (runDebugCmd) {
        int ret = std::system(schema.debugCmd.c_str());
        if (ret != 0) {
            IR_LOGW() << "ProExceptionDumpCallback: debug compile command failed (ret=" << ret << ")";
        }
    }

    return 0;
}

static int32_t ProExceptionDumpCallback(void* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo,
                                        uint32_t exceptionDumpSize, uint32_t* exceptionDumpRealSize,
                                        AdxExceptionDumpMode* mode)
{
    if (exceptionDumpInfo == nullptr || exceptionDumpRealSize == nullptr || mode == nullptr) {
        return 1;
    }
    ExceptionLaunchInfo runtimeArgs;
    if (!GetKernelExceptionArgs(exceptionInfo, runtimeArgs)) {
        *mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
        *exceptionDumpRealSize = 0U;
        IR_LOGW() << "ProExceptionDumpCallback: ACL could not resolve exception launch arguments; PyPTO dump skipped";
        return 0;
    }
    *mode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    *exceptionDumpRealSize = 0U;
    if (!ValidateRuntimeArgs(runtimeArgs)) {
        return 0;
    }
    std::vector<uint64_t> runtimeArgStorage(runtimeArgs.argsize / sizeof(uint64_t));
    const auto aclMemcpy = GetAclExceptionApi().memcpy;
    // ACL_MEMCPY_DEVICE_TO_HOST is 2; keep this C ABI boundary independent of ACL headers.
    const auto ret = aclMemcpy == nullptr ? -1 :
                                            aclMemcpy(runtimeArgStorage.data(), runtimeArgs.argsize,
                                                      runtimeArgs.argAddr, runtimeArgs.argsize, 2);
    if (ret != 0) {
        IR_LOGW() << "ProExceptionDumpCallback: copy runtime launch arguments failed (ret=" << ret << ")";
        return 0;
    }
    return ProExceptionDumpCallbackImpl(runtimeArgs, runtimeArgStorage.data(),
                                        static_cast<uint32_t>(runtimeArgStorage.size()), exceptionDumpInfo,
                                        exceptionDumpSize, exceptionDumpRealSize, mode);
}

extern "C" {

int32_t pro_register_exception_dump_callback()
{
    if (g_registered) {
        return 0;
    }
    auto regFunc = LoadAdumpRegFunc();
    if (regFunc == nullptr) {
        return -1;
    }
    int32_t ret = regFunc(ProExceptionDumpCallback);
    if (ret == 0) {
        g_registered = true;
    }
    return ret;
}

int32_t pro_register_dump_schema(const char* kernelName, int32_t numParams, const int32_t* kinds,
                                 const int32_t* tensorTypes, const int32_t* dataTypes, const int32_t* elementBits,
                                 const char* const* names, const char* const* dtypeNames, const int32_t* shapeCounts,
                                 const int64_t* flatShapes, const int32_t* flatDynamicSlots, const char* debugCmd)
{
    if (kernelName == nullptr || kernelName[0] == '\0' || numParams < 0 || numParams > MAX_TENSOR_NUM) {
        return 1;
    }
    if (numParams > 0 && (kinds == nullptr || tensorTypes == nullptr || dataTypes == nullptr ||
                          elementBits == nullptr || names == nullptr || dtypeNames == nullptr ||
                          shapeCounts == nullptr || flatShapes == nullptr || flatDynamicSlots == nullptr)) {
        return 1;
    }

    DumpKernelSchema schema;
    schema.params.reserve(static_cast<size_t>(numParams));
    for (int32_t i = 0; i < numParams; ++i) {
        if (kinds[i] < static_cast<int32_t>(KernelParamKind::TENSOR) ||
            kinds[i] > static_cast<int32_t>(KernelParamKind::WINDOW) || shapeCounts[i] < 0 ||
            shapeCounts[i] > MAX_SCHEMA_DIMS) {
            return 1;
        }
        DumpParamSchema param;
        param.name = names[i] == nullptr ? "" : names[i];
        param.dtypeName = dtypeNames[i] == nullptr ? "" : dtypeNames[i];
        param.kind = static_cast<KernelParamKind>(kinds[i]);
        param.tensorType = static_cast<AdxTensorType>(tensorTypes[i]);
        param.dataType = dataTypes[i];
        param.elementBits = elementBits[i];
        param.argSlot = static_cast<uint32_t>(i);
        for (int32_t dim = 0; dim < shapeCounts[i]; ++dim) {
            const size_t offset = static_cast<size_t>(i) * MAX_SCHEMA_DIMS + static_cast<size_t>(dim);
            param.shape.push_back(flatShapes[offset]);
            param.dynamicSlots.push_back(flatDynamicSlots[offset]);
        }
        schema.params.push_back(std::move(param));
    }
    if (debugCmd != nullptr) {
        schema.debugCmd = debugCmd;
    }

    std::lock_guard<std::mutex> lock(g_dumpSchemasMutex);
    g_dumpSchemas[std::string(kernelName)] = std::move(schema);
    return 0;
}

int32_t pro_unregister_dump_schema(const char* kernelName)
{
    if (kernelName == nullptr || kernelName[0] == '\0') {
        return 1;
    }
    std::lock_guard<std::mutex> lock(g_dumpSchemasMutex);
    g_dumpSchemas.erase(kernelName);
    return 0;
}

int32_t pro_set_dump_runtime_info(const char* kernelName, uint32_t blockDim, int32_t numSizes, const size_t* argSizes)
{
    if (kernelName == nullptr || kernelName[0] == '\0' || numSizes < 0 || (numSizes > 0 && argSizes == nullptr)) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(g_dumpSchemasMutex);
    auto it = g_dumpSchemas.find(kernelName);
    if (it == g_dumpSchemas.end()) {
        return 1;
    }
    size_t expectedSizes = 0U;
    for (const auto& param : it->second.params) {
        if (param.kind == KernelParamKind::PTR || param.kind == KernelParamKind::TILING) {
            ++expectedSizes;
        }
    }
    if (static_cast<size_t>(numSizes) != expectedSizes) {
        return 1;
    }
    it->second.blockDim = blockDim;
    size_t sizeIndex = 0U;
    for (auto& param : it->second.params) {
        if (param.kind == KernelParamKind::PTR || param.kind == KernelParamKind::TILING) {
            param.runtimeSize = argSizes[sizeIndex++];
        }
    }
    return 0;
}

#ifdef ENABLE_TESTS
extern "C" int32_t pro_test_exception_dump_callback_nullptr()
{
    return ProExceptionDumpCallback(nullptr, nullptr, 1, nullptr, nullptr);
}

extern "C" int32_t pro_test_exception_dump_callback_without_runtime_info(uint32_t* outRealSize, uint32_t* outMode)
{
    AdxExceptionDumpInfo dumpInfo = {};
    AdxExceptionDumpMode mode = AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE;
    uint32_t realSize = 1U;
    const int32_t ret = ProExceptionDumpCallback(nullptr, &dumpInfo, 1U, &realSize, &mode);
    *outRealSize = realSize;
    *outMode = static_cast<uint32_t>(mode);
    return ret;
}

extern "C" int32_t pro_test_exception_dump_callback_with_runtime_info(const char* kernelName, void* argAddr,
                                                                      uint32_t argsSize, void* bin, void** outArgAddr,
                                                                      uint32_t* outArgSize, void** outBin,
                                                                      char* outKernelName, uint32_t kernelNameBufSize,
                                                                      uint32_t* outExtraTensorNum, uint32_t* outMode)
{
    AdxExceptionDumpInfo dumpInfo = {};
    AdxExceptionDumpMode dumpMode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    uint32_t realDumpSize = 0;
    ExceptionLaunchInfo runtimeArgs{kernelName == nullptr ? "" : kernelName, argAddr, argsSize, bin};
    int32_t ret = 0;
    if (ValidateRuntimeArgs(runtimeArgs)) {
        std::vector<uint64_t> runtimeArgStorage(argsSize / sizeof(uint64_t));
        std::memcpy(runtimeArgStorage.data(), argAddr, argsSize);
        ret = ProExceptionDumpCallbackImpl(runtimeArgs, runtimeArgStorage.data(),
                                           static_cast<uint32_t>(runtimeArgStorage.size()), &dumpInfo, 1, &realDumpSize,
                                           &dumpMode);
    }
    if (outArgAddr != nullptr) {
        *outArgAddr = dumpInfo.argAddr;
    }
    if (outArgSize != nullptr) {
        *outArgSize = dumpInfo.argssize;
    }
    if (outBin != nullptr) {
        *outBin = dumpInfo.bin;
    }
    if (outKernelName != nullptr && kernelNameBufSize > 0) {
        snprintf(outKernelName, kernelNameBufSize, "%s", dumpInfo.kernelName);
    }
    if (outExtraTensorNum != nullptr) {
        *outExtraTensorNum = dumpInfo.extraTensorNum;
    }
    if (outMode != nullptr) {
        *outMode = static_cast<uint32_t>(dumpMode);
    }
    return ret;
}
#endif
}
