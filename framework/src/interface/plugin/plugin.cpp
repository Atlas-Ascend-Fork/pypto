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
 * \file plugin.cpp
 * \brief
 */

#include "plugin.h"

#include <dlfcn.h>
#include <utility>

#include "interface/plugin/plugin_name.h"
#include "tilefwk/tilefwk_log.h"

namespace npu::tile_fwk {

namespace {
using GetVersionEntry = const char* (*)();
} // namespace

SharedLibrary::SharedLibrary(std::string libName, std::string libVersion)
    : libName_(std::move(libName)), libVersion_(std::move(libVersion))
{}

SharedLibrary::~SharedLibrary() { Unload(); }

bool SharedLibrary::Load()
{
    if (handle_ != nullptr) {
        return true;
    }

    std::string library = "lib" + libName_ + ".so";
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        PYPTO_HOST_LOG_WITHOUT_MODULE(DLOG_WARN, "SharedLibrary: dlopen %s failed: %s", library.c_str(),
                                      error == nullptr ? "unknown" : error);
        return false;
    }

    auto getVersion = reinterpret_cast<GetVersionEntry>(dlsym(handle, PLUGIN_STR(PLUGIN_GET_VERSION_NAME)));
    if (getVersion == nullptr) {
        PYPTO_HOST_LOG_WITHOUT_MODULE(DLOG_WARN, "SharedLibrary: %s is missing %s", library.c_str(),
                                      PLUGIN_STR(PLUGIN_GET_VERSION_NAME));
        dlclose(handle);
        return false;
    }

    const char* actual = getVersion();
    if (actual == nullptr || std::string(actual) != libVersion_) {
        PYPTO_HOST_LOG_WITHOUT_MODULE(DLOG_WARN, "SharedLibrary: %s version %s != expected %s", library.c_str(),
                                      actual == nullptr ? "<null>" : actual, libVersion_.c_str());
        dlclose(handle);
        return false;
    }

    handle_ = handle;
    return true;
}

void SharedLibrary::Unload()
{
    if (handle_ != nullptr) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

void* SharedLibrary::GetSymbolAddress(const std::string& name) const
{
    return handle_ == nullptr ? nullptr : dlsym(handle_, name.c_str());
}

PluginManager& PluginManager::GetInstance()
{
    static PluginManager pluginManager;
    return pluginManager;
}

bool PluginManager::AddPlugin(const std::shared_ptr<PluginBase>& plugin)
{
    if (pluginDict_.count(plugin->GetName())) {
        return false;
    }
    pluginDict_[plugin->GetName()] = plugin;
    pluginListDict_[plugin->GetKind()].push_back(plugin);
    return true;
}

void PluginManager::ClearPlugin()
{
    pluginListDict_.clear();
    pluginDict_.clear();
}

} // namespace npu::tile_fwk
