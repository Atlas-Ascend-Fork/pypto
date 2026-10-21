/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <fstream>
#include <filesystem>

#define private public
#include "machine/runtime/runner/fed_prof_manager.h"
#undef private
#include "tilefwk/aicpu_common.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/runtime/memory_utils/memory_pool.h"
#include "interface/configs/config_manager.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace npu::tile_fwk::dynamic {
extern void FedProfPerCoreMixSyncEventDumpJsonList(json& mixSyncEventArray, const FedProfLeafTaskProf& taskStat,
                                                   const FedProfPerCoreProf* perCoreProf);
} // namespace npu::tile_fwk::dynamic

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DumpDevicePerfTest : public testing::Test {};

TEST_F(DumpDevicePerfTest, DumpDevTaskPerfData_EmptyPerfData)
{
    FedProfManager perf;
    perf.AicpuProfMaybeDumpSwimlane(false);
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, DumpDevTaskPerfData_EnvNotSet)
{
    FedProfManager perf;
    perf.AicpuProfMaybeDumpSwimlane(false);
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_NoEvents)
{
    FedProfLeafTaskProf stat{};
    stat.setEventNum = 0;
    stat.waitEventNum = 0;
    stat.setEventAddr = 0;
    stat.waitEventAddr = 0;
    std::vector<uint8_t> buf(256, 0);
    json result;
    FedProfPerCoreMixSyncEventDumpJsonList(result, stat, reinterpret_cast<FedProfPerCoreProf*>(buf.data()));
    EXPECT_TRUE(result.empty());
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_SetEventsOnly)
{
    std::vector<uint8_t> buf(1024, 0);
    uint64_t* setEventArea = reinterpret_cast<uint64_t*>(buf.data() + 64);
    setEventArea[0] = 100;
    setEventArea[1] = 0;
    setEventArea[2] = 300;
    FedProfLeafTaskProf stat{};
    stat.setEventNum = 3;
    stat.waitEventNum = 0;
    stat.setEventAddr = 64;
    stat.waitEventAddr = 512;
    json result;
    FedProfPerCoreMixSyncEventDumpJsonList(result, stat, reinterpret_cast<FedProfPerCoreProf*>(buf.data()));
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0]["type"], "CV_SYNC_SET");
    EXPECT_EQ(result[0]["idx"], 0);
    EXPECT_EQ(result[0]["time"], 100u);
    EXPECT_EQ(result[1]["idx"], 2);
    EXPECT_EQ(result[1]["time"], 300u);
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_WaitEventsOnly)
{
    std::vector<uint8_t> buf(1024, 0);
    uint64_t* waitEventArea = reinterpret_cast<uint64_t*>(buf.data() + 256);
    waitEventArea[0] = 200;
    waitEventArea[1] = 400;
    FedProfLeafTaskProf stat{};
    stat.setEventNum = 0;
    stat.waitEventNum = 2;
    stat.setEventAddr = 0;
    stat.waitEventAddr = 256;
    json result;
    FedProfPerCoreMixSyncEventDumpJsonList(result, stat, reinterpret_cast<FedProfPerCoreProf*>(buf.data()));
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0]["type"], "CV_SYNC_WAIT");
    EXPECT_EQ(result[0]["time"], 200u);
    EXPECT_EQ(result[1]["time"], 400u);
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_MixedAndSorted)
{
    std::vector<uint8_t> buf(2048, 0);
    uint64_t* setArea = reinterpret_cast<uint64_t*>(buf.data() + 128);
    uint64_t* waitArea = reinterpret_cast<uint64_t*>(buf.data() + 512);
    setArea[0] = 500;
    waitArea[0] = 100;
    waitArea[1] = 300;
    FedProfLeafTaskProf stat{};
    stat.setEventNum = 1;
    stat.waitEventNum = 2;
    stat.setEventAddr = 128;
    stat.waitEventAddr = 512;
    json result;
    FedProfPerCoreMixSyncEventDumpJsonList(result, stat, reinterpret_cast<FedProfPerCoreProf*>(buf.data()));
    EXPECT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0]["time"], 100u);
    EXPECT_EQ(result[0]["type"], "CV_SYNC_WAIT");
    EXPECT_EQ(result[1]["time"], 300u);
    EXPECT_EQ(result[1]["type"], "CV_SYNC_WAIT");
    EXPECT_EQ(result[2]["time"], 500u);
    EXPECT_EQ(result[2]["type"], "CV_SYNC_SET");
}

TEST_F(DumpDevicePerfTest, BuildSyncEventsJson_AllZeroTimestamps)
{
    std::vector<uint8_t> buf(1024, 0);
    FedProfLeafTaskProf stat{};
    stat.setEventNum = 3;
    stat.waitEventNum = 2;
    stat.setEventAddr = 0;
    stat.waitEventAddr = 256;
    json result;
    FedProfPerCoreMixSyncEventDumpJsonList(result, stat, reinterpret_cast<FedProfPerCoreProf*>(buf.data()));
    EXPECT_TRUE(result.empty());
}

TEST_F(DumpDevicePerfTest, DumpAicoreTaskExectInfo_Basic)
{
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 1;
    args.nrValidAic = 1;
    args.nrAicpu = 2;

    std::vector<uint8_t> buffers(3 * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE, 0);

    setenv("DUMP_DEVICE_PERF", "true", 1);
    FedProfManager perf;
    perf.args_ = args;
    uint64_t perCoreProfAddr = perf.DeviceDataAllocate(FED_PROF_PER_CORE_PROF);
    NormalizedRtMemcpy(dynamic::ValueToPtr(perCoreProfAddr), buffers.size(), buffers.data(), buffers.size(),
                       RtMemcpyKind::HOST_TO_DEVICE);
    perf.PerCoreProfDumpSwimlane();
    unsetenv("DUMP_DEVICE_PERF");
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, DumpAicoreTaskExectInfo_MultipleBlocks)
{
    FedProfDeviceArgs args{};
    args.nrAic = 2;
    args.nrAiv = 2;
    args.nrValidAic = 2;
    args.nrAicpu = 3;

    std::vector<uint8_t> buffers(5 * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE, 0);

    setenv("DUMP_DEVICE_PERF", "true", 1);
    FedProfManager perf;
    perf.args_ = args;
    uint64_t perCoreProfAddr = perf.DeviceDataAllocate(FED_PROF_PER_CORE_PROF);
    NormalizedRtMemcpy(dynamic::ValueToPtr(perCoreProfAddr), buffers.size(), buffers.data(), buffers.size(),
                       RtMemcpyKind::HOST_TO_DEVICE);
    perf.PerCoreProfDumpSwimlane();
    unsetenv("DUMP_DEVICE_PERF");
    SUCCEED();
}

TEST_F(DumpDevicePerfTest, DumpAicoreTaskExectInfo_WithSwimlaneFiles)
{
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 1;
    args.nrValidAic = 1;
    args.nrAicpu = 2;
    args.archInfo = ArchInfo::DAV_2201;

    std::string logFolder = npu::tile_fwk::config::LogTopFolder();
    std::filesystem::create_directories(logFolder);

    std::string programJsonPath = logFolder + "/program.json";
    std::string dynTopoPath = logFolder + "/dyn_topo.txt";

    std::ofstream programJson(programJsonPath);
    programJson << "{}";
    programJson.close();

    std::ofstream dynTopo(dynTopoPath);
    dynTopo << "dummy topo";
    dynTopo.close();

    std::vector<uint8_t> buffers(3 * FED_PROF_PER_CORE_PROF_ALIGNED_SIZE, 0);

    setenv("DUMP_DEVICE_PERF", "true", 1);
    FedProfManager perf;
    perf.args_ = args;
    uint64_t perCoreProfAddr = perf.DeviceDataAllocate(FED_PROF_PER_CORE_PROF);
    NormalizedRtMemcpy(dynamic::ValueToPtr(perCoreProfAddr), buffers.size(), buffers.data(), buffers.size(),
                       RtMemcpyKind::HOST_TO_DEVICE);
    perf.PerCoreProfDumpSwimlane();
    unsetenv("DUMP_DEVICE_PERF");

    std::filesystem::remove(programJsonPath);
    std::filesystem::remove(dynTopoPath);
    SUCCEED();
}
