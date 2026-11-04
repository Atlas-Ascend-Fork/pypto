/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/common/backend_registry.h"

#include <memory>
#include <string>
#include <utility>

#include "backend/common/backend.h"
#include "backend/common/soc.h"
#include "core/error.h"
#include "core/logging.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using npu::tile_fwk::ExternalError;

BackendRegistry& BackendRegistry::Instance()
{
    static BackendRegistry instance;
    return instance;
}

void BackendRegistry::Register(const std::string& typeName, CreateFunc func)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, registry_.find(typeName) == registry_.end())
        << "Backend type already registered: " << typeName;
    registry_[typeName] = std::move(func);
}

std::unique_ptr<Backend> BackendRegistry::Create(const std::string& typeName, const std::shared_ptr<const SoC>& soc)
{
    (void)typeName;
    (void)soc;
    PRO_CODEGEN_THROW(::pypto::ir::ValueError, npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR)
        << "Cannot create backend instances via registry - backends are singletons. "
        << "Use BackendCCE::Instance() instead.";
}

bool BackendRegistry::IsRegistered(const std::string& typeName) const
{
    return registry_.find(typeName) != registry_.end();
}

std::unique_ptr<Backend> CreateBackendFromRegistry(const std::string& typeName, const std::shared_ptr<const SoC>& soc)
{
    (void)typeName;
    (void)soc;
    PRO_CODEGEN_THROW(::pypto::ir::ValueError, npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR)
        << "Cannot create backend instances via registry - backends are singletons. "
        << "Use BackendCCE::Instance() instead.";
}

namespace {
bool RegisterBackendCCE()
{
    BackendRegistry::Instance().Register("CCE", [](const std::shared_ptr<const SoC>& /*unused*/) {
        PRO_CODEGEN_THROW(::pypto::ir::ValueError, npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR)
            << "Cannot create BackendCCE via registry - use BackendCCE::Instance()";
        return std::unique_ptr<Backend>(nullptr);
    });
    return true;
}

static bool backendCceRegistered = RegisterBackendCCE();
} // namespace

} // namespace backend
} // namespace pypto
