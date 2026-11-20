/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

#include "pypto_pro/backend/common/cann_runtime.h"

extern "C" {
int32_t pro_register_exception_dump_callback();
int32_t pro_register_dump_schema(const char* kernelName, int32_t numParams, const int32_t* kinds,
                                 const int32_t* tensorTypes, const int32_t* dataTypes, const int32_t* elementBits,
                                 const char* const* names, const char* const* dtypeNames, const int32_t* shapeCounts,
                                 const int64_t* flatShapes, const int32_t* flatDynamicSlots, const char* debugCmd);
int32_t pro_unregister_dump_schema(const char* kernelName);
int32_t pro_set_dump_runtime_info(const char* kernelName, uint32_t blockDim, int32_t numSizes, const size_t* argSizes);
int32_t pro_test_exception_dump_callback_nullptr();
int32_t pro_test_exception_dump_callback_without_runtime_info(uint32_t* outRealSize, uint32_t* outMode);
int32_t pro_test_exception_dump_callback_with_runtime_info(const char* kernelName, void* argAddr, uint32_t argsSize,
                                                           void* bin, void** outArgAddr, uint32_t* outArgSize,
                                                           void** outBin, char* outKernelName,
                                                           uint32_t kernelNameBufSize, uint32_t* outExtraTensorNum,
                                                           uint32_t* outMode, uint32_t dumpCapacity = 1U,
                                                           uint32_t* outRealSize = nullptr);
}

namespace {
void RegisterDynamicSchema(const char* kernelName, const char* debugCmd = "")
{
    int32_t kinds[] = {static_cast<int32_t>(pypto::backend::KernelParamKind::TENSOR),
                       static_cast<int32_t>(pypto::backend::KernelParamKind::SCALAR),
                       static_cast<int32_t>(pypto::backend::KernelParamKind::PTR)};
    int32_t tensorTypes[] = {0, 0, 0};
    int32_t dataTypes[] = {1, 9, 4};
    int32_t elementBits[] = {16, 64, 8};
    const char* names[] = {"input", "scale", "workspace"};
    const char* dtypeNames[] = {"DT_FP16", "DT_INT64", "DT_UINT8"};
    int32_t shapeCounts[] = {2, 0, 0};
    int64_t flatShapes[24] = {4, 0};
    int32_t flatDynamicSlots[24] = {-1, 3};
    ASSERT_EQ(pro_register_dump_schema(kernelName, 3, kinds, tensorTypes, dataTypes, elementBits, names, dtypeNames,
                                       shapeCounts, flatShapes, flatDynamicSlots, debugCmd),
              0);
    ASSERT_EQ(pro_set_dump_runtime_info(kernelName, 8, 0, nullptr), 1);
    size_t sizes[] = {64};
    ASSERT_EQ(pro_set_dump_runtime_info(kernelName, 8, 1, sizes), 0);
}
} // namespace

TEST(ExceptionDumpCallbackTest, RegisterCallbackReturnsResultCode)
{
    const int32_t ret = pro_register_exception_dump_callback();
    EXPECT_TRUE(ret == 0 || ret == -1);
}

TEST(ExceptionDumpCallbackTest, CallbackNullptrDumpInfoReturnsError)
{
    EXPECT_EQ(pro_test_exception_dump_callback_nullptr(), 1);
}

TEST(ExceptionDumpCallbackTest, CallbackWithoutRuntimeInfoProducesNoDump)
{
    uint32_t realSize = 1U;
    uint32_t mode = 1U;
    ASSERT_EQ(pro_test_exception_dump_callback_without_runtime_info(&realSize, &mode), 0);
    EXPECT_EQ(realSize, 0U);
    EXPECT_EQ(mode, 0U);
}

TEST(ExceptionDumpCallbackTest, RejectsInvalidSchemaAndRuntimeUpdates)
{
    const char kernelName[] = "PyPTO_InvalidSchemaKernel";
    int32_t kinds[] = {static_cast<int32_t>(pypto::backend::KernelParamKind::TENSOR)};
    int32_t tensorTypes[] = {0};
    int32_t dataTypes[] = {1};
    int32_t elementBits[] = {16};
    const char* names[] = {"input"};
    const char* dtypeNames[] = {"DT_FP16"};
    int32_t shapeCounts[] = {1};
    int64_t flatShapes[8] = {16};
    int32_t flatDynamicSlots[8] = {-1};
    const auto registerSchema = [&]() {
        return pro_register_dump_schema(kernelName, 1, kinds, tensorTypes, dataTypes, elementBits, names, dtypeNames,
                                        shapeCounts, flatShapes, flatDynamicSlots, "");
    };

    EXPECT_EQ(pro_register_dump_schema(nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr),
              1);
    EXPECT_EQ(pro_register_dump_schema(kernelName, -1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr),
              1);
    EXPECT_EQ(pro_register_dump_schema(kernelName, 1, nullptr, tensorTypes, dataTypes, elementBits, names, dtypeNames,
                                       shapeCounts, flatShapes, flatDynamicSlots, ""),
              1);
    kinds[0] = -1;
    EXPECT_EQ(registerSchema(), 1);
    kinds[0] = static_cast<int32_t>(pypto::backend::KernelParamKind::TENSOR);
    shapeCounts[0] = 9;
    EXPECT_EQ(registerSchema(), 1);
    shapeCounts[0] = 1;
    ASSERT_EQ(registerSchema(), 0);

    EXPECT_EQ(pro_set_dump_runtime_info(nullptr, 8, 0, nullptr), 1);
    EXPECT_EQ(pro_set_dump_runtime_info("PyPTO_MissingSchemaKernel", 8, 0, nullptr), 1);
    size_t size[] = {64};
    EXPECT_EQ(pro_set_dump_runtime_info(kernelName, 8, 1, size), 1);
    EXPECT_EQ(pro_set_dump_runtime_info(kernelName, 8, 0, nullptr), 0);
    EXPECT_EQ(pro_unregister_dump_schema(nullptr), 1);
    EXPECT_EQ(pro_unregister_dump_schema(kernelName), 0);
}

TEST(ExceptionDumpCallbackTest, CallbackWithoutSchemaDoesNotReturnStaleDumpInfo)
{
    const char kernelName[] = "PyPTO_NoSchemaKernel";
    uint64_t launchArgs[] = {0x1000U};
    uint32_t extraTensorNum = 99U;
    uint32_t mode = 99U;
    ASSERT_EQ(
        pro_test_exception_dump_callback_with_runtime_info(kernelName, launchArgs, sizeof(launchArgs), nullptr, nullptr,
                                                           nullptr, nullptr, nullptr, 0, &extraTensorNum, &mode),
        0);
    EXPECT_EQ(extraTensorNum, 0U);
    EXPECT_EQ(mode, 0U);
}

TEST(ExceptionDumpCallbackTest, UnregisterSchemaPreventsStaleDumpInfo)
{
    const char kernelName[] = "PyPTO_UnregisteredSchemaKernel";
    RegisterDynamicSchema(kernelName);
    EXPECT_EQ(pro_unregister_dump_schema(kernelName), 0);

    uint64_t launchArgs[] = {0x1000U, 17U, 0x2000U, 8U};
    uint32_t extraTensorNum = 99U;
    uint32_t mode = 99U;
    ASSERT_EQ(
        pro_test_exception_dump_callback_with_runtime_info(kernelName, launchArgs, sizeof(launchArgs), nullptr, nullptr,
                                                           nullptr, nullptr, nullptr, 0, &extraTensorNum, &mode),
        0);
    EXPECT_EQ(extraTensorNum, 0U);
    EXPECT_EQ(mode, 0U);
}

TEST(ExceptionDumpCallbackTest, CallbackReconstructsDynamicTensorFromRegisteredSchema)
{
    const char kernelName[] = "PyPTO_SchemaKernel";
    RegisterDynamicSchema(kernelName);

    uint64_t launchArgs[] = {0x1000U, 17U, 0x2000U, 8U};
    uint8_t kernelBinary[64] = {};
    void* outArgAddr = nullptr;
    uint32_t outArgSize = 0;
    void* outBin = nullptr;
    char outKernelName[1024] = {};
    uint32_t extraTensorNum = 0;
    uint32_t mode = 0;
    ASSERT_EQ(pro_test_exception_dump_callback_with_runtime_info(
                  kernelName, launchArgs, sizeof(launchArgs), kernelBinary, &outArgAddr, &outArgSize, &outBin,
                  outKernelName, sizeof(outKernelName), &extraTensorNum, &mode),
              0);

    EXPECT_EQ(outArgAddr, nullptr);
    EXPECT_EQ(outArgSize, 0U);
    EXPECT_EQ(outBin, nullptr);
    EXPECT_STREQ(outKernelName, kernelName);
    EXPECT_EQ(extraTensorNum, 2U);
    EXPECT_EQ(mode, 2U); // ADDITIONAL retains CANN's native all-core diagnostics.
}

TEST(ExceptionDumpCallbackTest, MultiCoreCapacitySupplementsNativeDiagnosticsAndRebuildsOnlyOnce)
{
    const std::string workPath = testing::TempDir();
    const std::string markerPath = workPath + "/pypto_exception_rebuild_count";
    std::remove(markerPath.c_str());
    const char kernelName[] = "PyPTO_MultiCoreSnapshotKernel";
    const std::string debugCmd = "printf x >> '" + markerPath + "'";
    RegisterDynamicSchema(kernelName, debugCmd.c_str());
    uint64_t launchArgs[] = {0x1000U, 17U, 0x2000U, 8U};
    uint8_t kernelBinary[64] = {};

    for (uint32_t capacity : {0U, 56U, 1U}) {
        uint32_t realSize = 99U;
        uint32_t extraTensorNum = 99U;
        uint32_t mode = 99U;
        void* outArgAddr = launchArgs;
        uint32_t outArgSize = sizeof(launchArgs);
        void* outBin = kernelBinary;
        ASSERT_EQ(pro_test_exception_dump_callback_with_runtime_info(
                      kernelName, launchArgs, sizeof(launchArgs), kernelBinary, &outArgAddr, &outArgSize, &outBin,
                      nullptr, 0, &extraTensorNum, &mode, capacity, &realSize),
                  0);
        EXPECT_EQ(realSize, capacity == 0U ? 0U : 1U);
        EXPECT_EQ(extraTensorNum, capacity == 0U ? 0U : 2U);
        EXPECT_EQ(mode, capacity == 0U ? 0U : 2U);
        EXPECT_EQ(outArgAddr, nullptr);
        EXPECT_EQ(outArgSize, 0U);
        EXPECT_EQ(outBin, nullptr);
        if (capacity == 0U) {
            FILE* marker = std::fopen(markerPath.c_str(), "rb");
            EXPECT_EQ(marker, nullptr);
            if (marker != nullptr) {
                std::fclose(marker);
            }
        }
    }
    FILE* marker = std::fopen(markerPath.c_str(), "rb");
    ASSERT_NE(marker, nullptr);
    EXPECT_EQ(std::fgetc(marker), 'x');
    EXPECT_EQ(std::fgetc(marker), EOF);
    std::fclose(marker);
    std::remove(markerPath.c_str());
    EXPECT_EQ(pro_unregister_dump_schema(kernelName), 0);
}

TEST(ExceptionDumpCallbackTest, CallbackRejectsInvalidRuntimeArgumentsAndTensorExtents)
{
    const char kernelName[] = "PyPTO_InvalidRuntimeArgsKernel";
    RegisterDynamicSchema(kernelName);
    uint64_t launchArgs[] = {0x1000U, 17U, 0x2000U, 8U};
    const auto check = [&](void* args, uint32_t size, uint32_t expectedTensors, uint32_t expectedMode) {
        uint32_t extraTensorNum = 99U;
        uint32_t mode = 99U;
        EXPECT_EQ(pro_test_exception_dump_callback_with_runtime_info(kernelName, args, size, nullptr, nullptr, nullptr,
                                                                     nullptr, nullptr, 0, &extraTensorNum, &mode),
                  0);
        EXPECT_EQ(extraTensorNum, expectedTensors);
        EXPECT_EQ(mode, expectedMode);
    };

    check(nullptr, sizeof(launchArgs), 0U, 0U);
    check(launchArgs, 0U, 0U, 0U);
    check(launchArgs, 7U, 0U, 0U);
    check(launchArgs, 1024U * 1024U + 8U, 0U, 0U);
    check(launchArgs, sizeof(launchArgs) - sizeof(uint64_t), 1U, 2U); // Missing dynamic extent.
    check(launchArgs, 2U * sizeof(uint64_t), 0U, 2U);                 // Both Tensor and Ptr slots are incomplete.
    launchArgs[3] = 0U;
    check(launchArgs, sizeof(launchArgs), 1U, 2U);
    launchArgs[3] = static_cast<uint64_t>(INT64_MAX);
    check(launchArgs, sizeof(launchArgs), 1U, 2U); // 4 * extent overflows uint64_t.
    launchArgs[3] = 8U;
    launchArgs[0] = 0U;
    check(launchArgs, sizeof(launchArgs), 1U, 2U);
    launchArgs[2] = 0U;
    check(launchArgs, sizeof(launchArgs), 0U, 2U);
    EXPECT_EQ(pro_unregister_dump_schema(kernelName), 0);
}

TEST(ExceptionDumpCallbackTest, CallbackMatchesMangledKernelName)
{
    const char kernelName[] = "oob_load_tile_kernel";
    const char runtimeName[] = "_Z20oob_load_tile_kernelPDhPfS_S0_llllllll_mix_aiv";
    RegisterDynamicSchema(kernelName);
    uint64_t launchArgs[] = {0x1000U, 17U, 0x2000U, 8U};
    char outKernelName[1024] = {};
    uint32_t extraTensorNum = 0U;
    uint32_t mode = 0U;
    ASSERT_EQ(pro_test_exception_dump_callback_with_runtime_info(runtimeName, launchArgs, sizeof(launchArgs), nullptr,
                                                                 nullptr, nullptr, nullptr, outKernelName,
                                                                 sizeof(outKernelName), &extraTensorNum, &mode),
              0);
    EXPECT_STREQ(outKernelName, kernelName);
    EXPECT_EQ(extraTensorNum, 2U);
    EXPECT_EQ(mode, 2U);
}

TEST(ExceptionDumpCallbackTest, CallbackWritesLaunchMetadataFromRuntimeArguments)
{
    const std::string workPath = testing::TempDir();
    ASSERT_NE(::setenv("ASCEND_WORK_PATH", workPath.c_str(), 1), -1);
    ASSERT_NE(::setenv("TILE_FWK_DEVICE_ID", "0", 1), -1);
    const std::string dumpDir = workPath + "/extra-info/data-dump/0";
    for (const std::string& dir : {workPath + "/extra-info", workPath + "/extra-info/data-dump", dumpDir}) {
        ASSERT_TRUE(::mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) << "mkdir failed: " << dir;
    }

    const char kernelName[] = "PyPTO_MetadataKernel";
    RegisterDynamicSchema(kernelName);
    size_t workspaceSize[] = {128};
    ASSERT_EQ(pro_set_dump_runtime_info(kernelName, 16, 1, workspaceSize), 0);
    uint64_t launchArgs[] = {0x1000U, 17U, 0x2000U, 8U};
    void* outArgAddr = nullptr;
    uint32_t outArgSize = 0U;
    uint32_t extraTensorNum = 0U;
    uint32_t mode = 0U;
    ASSERT_EQ(pro_test_exception_dump_callback_with_runtime_info(kernelName, launchArgs, sizeof(launchArgs), nullptr,
                                                                 &outArgAddr, &outArgSize, nullptr, nullptr, 0,
                                                                 &extraTensorNum, &mode),
              0);
    EXPECT_EQ(outArgAddr, nullptr);
    EXPECT_EQ(outArgSize, 0U);
    EXPECT_EQ(extraTensorNum, 2U);
    EXPECT_EQ(mode, 2U);

    const std::string metaPath = dumpDir + "/PyPTO_MetadataKernel_launch_args.json";
    FILE* file = std::fopen(metaPath.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    char buffer[1024] = {};
    const size_t size = std::fread(buffer, 1, sizeof(buffer) - 1U, file);
    std::fclose(file);
    const std::string metadata(buffer, size);
    EXPECT_NE(metadata.find("\"block_dim\":16"), std::string::npos);
    EXPECT_NE(
        metadata.find(
            "\"name\":\"input\",\"kind\":\"ptr\",\"dtype\":\"DT_FP16\",\"addr\":4096,\"size\":64,\"shape\":[4,8]"),
        std::string::npos);
    EXPECT_NE(
        metadata.find(
            "\"name\":\"workspace\",\"kind\":\"ptr\",\"dtype\":\"DT_UINT8\",\"addr\":8192,\"size\":128,\"shape\":[]"),
        std::string::npos);
    EXPECT_NE(metadata.find("\"value\":17"), std::string::npos);

    ::unsetenv("ASCEND_WORK_PATH");
    ::unsetenv("TILE_FWK_DEVICE_ID");
}

TEST(ExceptionDumpCallbackTest, WindowDoesNotPreventTensorDumpAndNonfiniteScalarMetadata)
{
    const std::string workPath = testing::TempDir();
    ASSERT_NE(::setenv("ASCEND_WORK_PATH", workPath.c_str(), 1), -1);
    ASSERT_NE(::setenv("TILE_FWK_DEVICE_ID", "0", 1), -1);
    const std::string dumpDir = workPath + "/extra-info/data-dump/0";
    for (const std::string& dir : {workPath + "/extra-info", workPath + "/extra-info/data-dump", dumpDir}) {
        ASSERT_TRUE(::mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) << "mkdir failed: " << dir;
    }

    const char kernelName[] = "PyPTO_WindowSchemaKernel";
    int32_t kinds[] = {static_cast<int32_t>(pypto::backend::KernelParamKind::WINDOW),
                       static_cast<int32_t>(pypto::backend::KernelParamKind::TENSOR),
                       static_cast<int32_t>(pypto::backend::KernelParamKind::SCALAR),
                       static_cast<int32_t>(pypto::backend::KernelParamKind::SCALAR),
                       static_cast<int32_t>(pypto::backend::KernelParamKind::SCALAR)};
    int32_t tensorTypes[] = {0, 0, 0, 0, 0};
    int32_t dataTypes[] = {4, 1, 0, 0, 0}; // uint8, float16, float32, float32, float32
    int32_t elementBits[] = {8, 16, 32, 32, 32};
    const char* names[] = {"window", "input", "nonfinite", "precise", "negative_zero"};
    const char* dtypeNames[] = {"DT_UINT8", "DT_FP16", "DT_FP32", "DT_FP32", "DT_FP32"};
    int32_t shapeCounts[] = {0, 1, 0, 0, 0};
    int64_t flatShapes[40] = {};
    flatShapes[8] = 16;
    int32_t flatDynamicSlots[40] = {};
    ASSERT_EQ(pro_register_dump_schema(kernelName, 5, kinds, tensorTypes, dataTypes, elementBits, names, dtypeNames,
                                       shapeCounts, flatShapes, flatDynamicSlots, ""),
              0);
    uint64_t launchArgs[] = {0x3000U, 0x1000U, 0x7fc00000U, 0x3f9e0652U, 0x80000000U}; // NaN, 1.23456788f, -0.0f
    uint32_t extraTensorNum = 0U;
    ASSERT_EQ(
        pro_test_exception_dump_callback_with_runtime_info(kernelName, launchArgs, sizeof(launchArgs), nullptr, nullptr,
                                                           nullptr, nullptr, nullptr, 0, &extraTensorNum, nullptr),
        0);
    EXPECT_EQ(extraTensorNum, 1U);

    const std::string metaPath = dumpDir + "/PyPTO_WindowSchemaKernel_launch_args.json";
    FILE* file = std::fopen(metaPath.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    char buffer[1024] = {};
    const size_t size = std::fread(buffer, 1, sizeof(buffer) - 1U, file);
    std::fclose(file);
    const std::string metadata(buffer, size);
    EXPECT_NE(metadata.find("\"kind\":\"window\""), std::string::npos);
    EXPECT_NE(metadata.find("\"value\":NaN"), std::string::npos);
    EXPECT_NE(metadata.find("\"value\":1.23456788"), std::string::npos);
    EXPECT_NE(metadata.find("\"name\":\"negative_zero\",\"kind\":\"scalar\",\"value\":-0."), std::string::npos);

    ::unsetenv("ASCEND_WORK_PATH");
    ::unsetenv("TILE_FWK_DEVICE_ID");
}
