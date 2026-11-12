/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <vector>

#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/runner/runtime_agent.h"

using namespace npu::tile_fwk;

class RuntimeAgentNpuTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto ret = AclInit(nullptr);
        if (ret != ACLRT_SUCCESS && ret != ACLRT_ERROR_REPEAT_INITIALIZE) {
            GTEST_SKIP() << "AclInit failed";
        }
        ret = AclRtSetDevice(0);
        if (ret != ACLRT_SUCCESS) {
            GTEST_SKIP() << "AclRtSetDevice failed";
        }
    }

    void TearDown() override
    {
        AclRtResetDevice(0);
        AclFinalize();
    }
};

TEST_F(RuntimeAgentNpuTest, GetAicoreRegInfo_ReturnsValidData)
{
    RuntimeAgent agent;
    std::vector<int64_t> aic, aiv;
    int ret = agent.GetAicoreRegInfo(aic, aiv, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_FALSE(aic.empty());
    EXPECT_FALSE(aiv.empty());
}

TEST_F(RuntimeAgentNpuTest, GetAicoreRegInfoForDAVID_SkipsOnOtherArch)
{
    std::vector<int64_t> regs, regsPmu;
    RuntimeAgent::GetAicoreRegInfoForDAVID(regs, regsPmu, ArchInfo::DAV_2201);
    EXPECT_TRUE(regs.empty());
    EXPECT_TRUE(regsPmu.empty());
}
