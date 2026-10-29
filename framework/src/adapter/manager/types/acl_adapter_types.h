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
 * \file acl_adapter_types.h
 * \brief
 */

#pragma once

#include <string>
#include <map>

namespace npu::tile_fwk {
enum class AclFunc {
    Init = 0,
    Finalize,
    RtMemcpy,
    RtMemset,
    RtSetDevice,
    RtResetDevice,
    RtCreateEvent,
    RtRecordEvent,
    RtCreateEventExWithFlag,
    RtStreamWaitEvent,
    RtGetStreamResLimit,
    RtGetStreamAttribute,
    RtSetStreamAttribute,
    RtCacheLastTaskOpInfo,
    RtSetExceptionInfoCallback,
    MdlRICaptureGetInfo,
    MdlRICaptureThreadExchangeMode,
    RtMalloc,
    RtCreateStream,
    RtSynchronizeStream,
    RtFree,
    RtDestroyStream,
    RtGetDevice,
    RtGetDeviceInfo,
    BinaryLoadFromFile,
    BinaryGetFunction,
    SysGetVersionStr,
    Bottom
};

const std::string kAclLibName = "libascendcl.so";
const std::map<AclFunc, std::string> kAclFuncStrMap{
    {AclFunc::Init, "aclInit"},
    {AclFunc::Finalize, "aclFinalize"},
    {AclFunc::RtMemcpy, "aclrtMemcpy"},
    {AclFunc::RtMemset, "aclrtMemset"},
    {AclFunc::RtSetDevice, "aclrtSetDevice"},
    {AclFunc::RtResetDevice, "aclrtResetDevice"},
    {AclFunc::RtCreateEvent, "aclrtCreateEvent"},
    {AclFunc::RtRecordEvent, "aclrtRecordEvent"},
    {AclFunc::RtCreateEventExWithFlag, "aclrtCreateEventExWithFlag"},
    {AclFunc::RtStreamWaitEvent, "aclrtStreamWaitEvent"},
    {AclFunc::RtGetStreamResLimit, "aclrtGetStreamResLimit"},
    {AclFunc::RtGetStreamAttribute, "aclrtGetStreamAttribute"},
    {AclFunc::RtSetStreamAttribute, "aclrtSetStreamAttribute"},
    {AclFunc::RtCacheLastTaskOpInfo, "aclrtCacheLastTaskOpInfo"},
    {AclFunc::RtSetExceptionInfoCallback, "aclrtSetExceptionInfoCallback"},
    {AclFunc::MdlRICaptureGetInfo, "aclmdlRICaptureGetInfo"},
    {AclFunc::MdlRICaptureThreadExchangeMode, "aclmdlRICaptureThreadExchangeMode"},
    {AclFunc::RtMalloc, "aclrtMalloc"},
    {AclFunc::RtCreateStream, "aclrtCreateStream"},
    {AclFunc::RtSynchronizeStream, "aclrtSynchronizeStream"},
    {AclFunc::RtFree, "aclrtFree"},
    {AclFunc::RtDestroyStream, "aclrtDestroyStream"},
    {AclFunc::RtGetDevice, "aclrtGetDevice"},
    {AclFunc::RtGetDeviceInfo, "aclrtGetDeviceInfo"},
    {AclFunc::BinaryLoadFromFile, "aclrtBinaryLoadFromFile"},
    {AclFunc::BinaryGetFunction, "aclrtBinaryGetFunction"},
    {AclFunc::SysGetVersionStr, "aclsysGetVersionStr"}};

enum class AclNewFunc { NoBlockingLaunchBegin, NoBlockingLaunchEnd, Bottom };

const std::string newAclLibName = "libacl_rt.so";
const std::map<AclNewFunc, std::string> aclNewFuncStrMap{
    {AclNewFunc::NoBlockingLaunchBegin, "aclrtNonBlockingLaunchBegin"},
    {AclNewFunc::NoBlockingLaunchEnd, "aclrtNonBlockingLaunchEnd"},
};
} // namespace npu::tile_fwk
