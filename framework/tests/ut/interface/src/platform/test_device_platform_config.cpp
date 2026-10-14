/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_device_platform_config.cpp
 * \brief Test that device_platform config controls GetMemoryLimit data source
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"
#define private public
#include "platform/parser/platform_parser.h"
#include "platform/parser/simulation_platform/simulation_platform.h"

using namespace npu::tile_fwk;

const std::string version = "version";
const std::string aiCoreSpec = "AICoreSpec";
const std::string ubSize = "ub_size";
const std::string l0cSize = "l0_c_size";
const std::string l1Size = "l1_size";
const std::string l0aSize = "l0_a_size";
const std::string l0aMxSize = "l0_a_mx_size";
const std::string l0bMxSize = "l0_b_mx_size";
const std::string socVersionInfo = "SoC_version";
const std::string npuArchInfo = "NpuArch";

class TestDevicePlatformConfig : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
};

// ==================== StringToDPlatform Tests ====================

TEST_F(TestDevicePlatformConfig, StringToDPlatformAllMappings)
{
    EXPECT_EQ(StringToDPlatform("Ascend910B1"), DPlatform::ASCEND_910B1);
    EXPECT_EQ(StringToDPlatform("Ascend910B2"), DPlatform::ASCEND_910B2);
    EXPECT_EQ(StringToDPlatform("Ascend910B3"), DPlatform::ASCEND_910B3);
    EXPECT_EQ(StringToDPlatform("Ascend910B4"), DPlatform::ASCEND_910B4);
    EXPECT_EQ(StringToDPlatform("Ascend910_9363"), DPlatform::ASCEND_910_9363);
    EXPECT_EQ(StringToDPlatform("Ascend950DT_9572"), DPlatform::ASCEND_950DT_9572);
    EXPECT_EQ(StringToDPlatform("Ascend950PR_9579"), DPlatform::ASCEND_950PR_9579);
    EXPECT_EQ(StringToDPlatform("Ascend950DT_9581"), DPlatform::ASCEND_950DT_9581);
    EXPECT_EQ(StringToDPlatform("Ascend950DT_9581X"), DPlatform::ASCEND_950DT_9581X);
    EXPECT_EQ(StringToDPlatform("Ascend950DT_9582"), DPlatform::ASCEND_950DT_9582);
    EXPECT_EQ(StringToDPlatform("Ascend950PR_9589"), DPlatform::ASCEND_950PR_9589);
    EXPECT_EQ(StringToDPlatform("Kirin9030"), DPlatform::KIRIN_9030);
    EXPECT_EQ(StringToDPlatform("KirinX90"), DPlatform::KIRIN_X90);
}

TEST_F(TestDevicePlatformConfig, StringToDPlatformUnknown)
{
    EXPECT_EQ(StringToDPlatform("INVALID_PLATFORM"), DPlatform::UNKNOWN_DEVICE);
    EXPECT_EQ(StringToDPlatform(""), DPlatform::UNKNOWN_DEVICE);
    EXPECT_EQ(StringToDPlatform("ascend_910b2"), DPlatform::UNKNOWN_DEVICE);
    // ASCEND_910C4 was removed from DPlatform, it must resolve to UNKNOWN (not a default fallback).
    EXPECT_EQ(StringToDPlatform("ASCEND_910C4"), DPlatform::UNKNOWN_DEVICE);
}

TEST_F(TestDevicePlatformConfig, StringToDPlatformLegacyFormatCompat)
{
    EXPECT_EQ(StringToDPlatform("ASCEND_910B2"), DPlatform::ASCEND_910B2);
    EXPECT_EQ(StringToDPlatform("ASCEND_950PR_9579"), DPlatform::ASCEND_950PR_9579);
    EXPECT_EQ(StringToDPlatform("KIRIN_9030"), DPlatform::KIRIN_9030);
    EXPECT_EQ(StringToDPlatform("ASCEND_910_9363"), DPlatform::ASCEND_910_9363);
    EXPECT_EQ(StringToDPlatform("ASCEND_950DT_9581X"), DPlatform::ASCEND_950DT_9581X);
    // Legacy enum values whose names did not match the actual chip bin.
    EXPECT_EQ(StringToDPlatform("ASCEND_950DT_9579"), DPlatform::ASCEND_950DT_9572);
    EXPECT_EQ(StringToDPlatform("ASCEND_950PR_9582"), DPlatform::ASCEND_950PR_9589);
}

TEST_F(TestDevicePlatformConfig, DPlatformToNPUArchMapping)
{
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_910B1), NPUArch::DAV_2201);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_910B2), NPUArch::DAV_2201);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_910B3), NPUArch::DAV_2201);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_910B4), NPUArch::DAV_2201);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_910_9363), NPUArch::DAV_2201);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_950DT_9572), NPUArch::DAV_3510);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_950PR_9579), NPUArch::DAV_3510);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_950DT_9581), NPUArch::DAV_3510);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_950DT_9581X), NPUArch::DAV_3510);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_950DT_9582), NPUArch::DAV_3510);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::ASCEND_950PR_9589), NPUArch::DAV_3510);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::KIRIN_9030), NPUArch::DAV_3113);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::KIRIN_X90), NPUArch::DAV_3003);
    EXPECT_EQ(DPlatformToNPUArch(DPlatform::UNKNOWN_DEVICE), NPUArch::DAV_UNKNOWN);
}

TEST_F(TestDevicePlatformConfig, ConfigGetDevicePlatformChain)
{
    config::SetPlatformConfig("device_platform", std::string("Ascend950PR_9579"));
    DPlatform platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950PR_9579);

    config::SetPlatformConfig("device_platform", std::string("Ascend950DT_9572"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9572);

    config::SetPlatformConfig("device_platform", std::string("Ascend950DT_9581"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9581);

    config::SetPlatformConfig("device_platform", std::string("Ascend950DT_9581X"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9581X);

    config::SetPlatformConfig("device_platform", std::string("Ascend950DT_9582"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9582);

    config::SetPlatformConfig("device_platform", std::string("Ascend950PR_9589"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950PR_9589);

    config::SetPlatformConfig("device_platform", std::string("Ascend910_9363"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_910_9363);

    config::SetPlatformConfig("device_platform", std::string("Ascend910B2"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_910B2);

    config::SetPlatformConfig("device_platform", std::string("Kirin9030"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::KIRIN_9030);

    // Deprecated device_platform formats resolve to the same DPlatform through the in-memory config chain.
    config::SetPlatformConfig("device_platform", std::string("ASCEND_910B2"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_910B2);

    config::SetPlatformConfig("device_platform", std::string("ASCEND_950DT_9579"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9572);

    config::SetPlatformConfig("device_platform", std::string("ASCEND_950PR_9582"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950PR_9589);

    config::SetPlatformConfig("device_platform", std::string("ASCEND_950DT_9581X"));
    platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950DT_9581X);
}

TEST_F(TestDevicePlatformConfig, ConfigGetDevicePlatformUnknown)
{
    config::SetPlatformConfig("device_platform", std::string("INVALID_PLATFORM"));
    DPlatform platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::UNKNOWN_DEVICE);
}

// ==================== SimulationPlatform::GetDevicePlatform Tests ====================

TEST_F(TestDevicePlatformConfig, GetDevicePlatformConfigFileNotFound)
{
    // The shipped tile_fwk_config.json keeps the legacy "ASCEND_910B2" format; GetDevicePlatform()
    // normalizes it to the standard soc_version via the legacy conversion layer.
    std::string result = SimulationPlatform::GetDevicePlatform();
    EXPECT_EQ(result, "Ascend910B2");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsWithConfigSocVersion)
{
    config::SetPlatformConfig("device_platform", std::string("Ascend950PR_9579"));
    DPlatform platform = config::GetDevicePlatform();
    EXPECT_EQ(platform, DPlatform::ASCEND_950PR_9579);

    INIParser parser("Ascend950PR_9579");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew950DT9581Config)
{
    INIParser parser("Ascend950DT_9581");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend950DT_9581");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew950DT9582Config)
{
    INIParser parser("Ascend950DT_9582");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend950DT_9582");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew950DT9572Config)
{
    INIParser parser("Ascend950DT_9572");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend950DT_9572");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew950PR9589Config)
{
    INIParser parser("Ascend950PR_9589");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend950PR_9589");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew950DT9581XConfig)
{
    INIParser parser("Ascend950DT_9581X");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 253952UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 262144UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "3510");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend950DT_9581X");
}

TEST_F(TestDevicePlatformConfig, INIParserLoadsNew9109363Config)
{
    INIParser parser("Ascend910_9363");

    size_t memoryLimit;
    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 196608UL);

    EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit));
    EXPECT_EQ(memoryLimit, 131072UL);

    std::string archType;
    EXPECT_TRUE(parser.GetStringVal(version, npuArchInfo, archType));
    EXPECT_EQ(archType, "2201");

    std::string socVersion;
    EXPECT_TRUE(parser.GetStringVal(version, socVersionInfo, socVersion));
    EXPECT_EQ(socVersion, "Ascend910_9363");
}

TEST_F(TestDevicePlatformConfig, DifferentPlatformDifferentMemory)
{
    INIParser parser910B1("Ascend910B1");
    INIParser parser950PR("Ascend950PR_9579");
    INIParser parserKirin("Kirin9030");

    size_t ub910B1, ub950PR, ubKirin;
    EXPECT_TRUE(parser910B1.GetSizeVal(aiCoreSpec, ubSize, ub910B1));
    EXPECT_TRUE(parser950PR.GetSizeVal(aiCoreSpec, ubSize, ub950PR));
    EXPECT_TRUE(parserKirin.GetSizeVal(aiCoreSpec, ubSize, ubKirin));

    EXPECT_EQ(ub910B1, 196608UL);
    EXPECT_EQ(ub950PR, 253952UL);
    EXPECT_EQ(ubKirin, 131072UL);

    EXPECT_NE(ub910B1, ub950PR);
    EXPECT_NE(ub910B1, ubKirin);

    size_t l0c910B1, l0c950PR, l0cKirin;
    EXPECT_TRUE(parser910B1.GetSizeVal(aiCoreSpec, l0cSize, l0c910B1));
    EXPECT_TRUE(parser950PR.GetSizeVal(aiCoreSpec, l0cSize, l0c950PR));
    EXPECT_TRUE(parserKirin.GetSizeVal(aiCoreSpec, l0cSize, l0cKirin));

    EXPECT_EQ(l0c910B1, 131072UL);
    EXPECT_EQ(l0c950PR, 262144UL);
    EXPECT_EQ(l0cKirin, 65536UL);
}

// L0A/L0B MX scale buffer sizes are only configured for the Ascend950 series (placeholder for future use).
TEST_F(TestDevicePlatformConfig, L0MxSizeOnlyConfiguredFor950Series)
{
    const std::string mxSocVersions[] = {
        "Ascend950DT_9572", "Ascend950DT_9581", "Ascend950DT_9581X",
        "Ascend950DT_9582", "Ascend950PR_9579", "Ascend950PR_9589",
    };
    for (const auto& socVersion : mxSocVersions) {
        INIParser parser(socVersion);
        size_t memoryLimit;
        EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0aMxSize, memoryLimit)) << socVersion;
        EXPECT_EQ(memoryLimit, 4096UL) << socVersion;
        EXPECT_TRUE(parser.GetSizeVal(aiCoreSpec, l0bMxSize, memoryLimit)) << socVersion;
        EXPECT_EQ(memoryLimit, 4096UL) << socVersion;
    }

    const std::string nonMxSocVersions[] = {
        "Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4", "Ascend910_9363", "Kirin9030", "KirinX90",
    };
    for (const auto& socVersion : nonMxSocVersions) {
        INIParser parser(socVersion);
        size_t memoryLimit;
        EXPECT_FALSE(parser.GetSizeVal(aiCoreSpec, l0aMxSize, memoryLimit)) << socVersion;
        EXPECT_FALSE(parser.GetSizeVal(aiCoreSpec, l0bMxSize, memoryLimit)) << socVersion;
    }
}

// SetMemoryLimit uses l0_a_mx_size/l0_b_mx_size from ini when present, and falls back to default otherwise.
TEST_F(TestDevicePlatformConfig, SetMemoryLimitL0MxSizeIniOrDefault)
{
    INIParser parser950("Ascend950PR_9579");
    Platform::Instance().SetMemoryLimit(parser950);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0AMX), 4096UL);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0BMX), 4096UL);

    INIParser parser910("Ascend910B1");
    Platform::Instance().SetMemoryLimit(parser910);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0AMX), 2048UL);
    EXPECT_EQ(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0BMX), 2048UL);

    Platform::Instance().SetMemoryLimit(INIParser());
}

TEST_F(TestDevicePlatformConfig, GetMemoryLimitForArchReadsSpecialBufferSizes)
{
    EXPECT_EQ(GetMemoryLimitForArch(NPUArch::DAV_3510, kMemorySpaceScaling), 4096UL);
    EXPECT_EQ(GetMemoryLimitForArch(NPUArch::DAV_3510, kMemorySpaceBias), 4096UL);
    EXPECT_EQ(GetMemoryLimitForArch(NPUArch::DAV_3510, kMemorySpaceScaleLeft), 4096UL);
    EXPECT_EQ(GetMemoryLimitForArch(NPUArch::DAV_3510, kMemorySpaceScaleRight), 4096UL);
}

// HasKey probes optional keys quietly, without reporting errors for missing section/key.
TEST_F(TestDevicePlatformConfig, HasKeyProbesOptionalSpecQuietly)
{
    INIParser parser950("Ascend950PR_9579");
    EXPECT_TRUE(parser950.HasKey(aiCoreSpec, l0aMxSize));
    EXPECT_TRUE(parser950.HasKey(aiCoreSpec, l0bMxSize));
    EXPECT_TRUE(parser950.HasKey(aiCoreSpec, l0aSize));

    INIParser parser910("Ascend910B1");
    EXPECT_FALSE(parser910.HasKey(aiCoreSpec, l0aMxSize));
    EXPECT_FALSE(parser910.HasKey(aiCoreSpec, l0bMxSize));
    EXPECT_TRUE(parser910.HasKey(aiCoreSpec, l0aSize));
    EXPECT_FALSE(parser910.HasKey(aiCoreSpec, "no_such_key"));
    EXPECT_FALSE(parser910.HasKey("NoSuchSection", l0aMxSize));
}
