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
 * \file plugin.h
 * \brief
 */

#include <cstdint>
#include <functional>
#include <string>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir/program.h"

namespace npu::tile_fwk {

enum Status : uint32_t;

enum class PluginKind : int {
    CODEGEN_SRC,
    PASS_PGO_VF,
};

class PluginBase {
public:
    PluginBase(PluginKind kind, const std::string& name) : kind_(kind), name_(name) {}
    virtual ~PluginBase() = default;

    PluginKind GetKind() { return kind_; }
    const std::string& GetName() const { return name_; }

    virtual bool Load() { return true; }
    virtual void Unload() {}

private:
    PluginKind kind_;
    std::string name_;
};

class SharedLibrary {
public:
    SharedLibrary(std::string libName, std::string libVersion);
    ~SharedLibrary();

    SharedLibrary(const SharedLibrary&) = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

    bool Valid() const { return handle_ != nullptr; }

    bool Load();
    void Unload();

    void* GetSymbolAddress(const std::string& name) const;

private:
    std::string libName_;
    std::string libVersion_;
    void* handle_{nullptr};
};

template <PluginKind tkind, typename TRetType, typename... TArgType>
class Plugin : public PluginBase {
public:
    static constexpr PluginKind kind = tkind;
    using RetType = TRetType;
    static_assert(!std::is_void_v<RetType>, "Plugin return type must not be void");
    typedef std::function<TRetType(TArgType...)> EntryType;

    Plugin(const std::string& name, EntryType entryHandler)
        : PluginBase(kind, name), entryHandler_(std::move(entryHandler))
    {}

    TRetType Call(TArgType... args) { return entryHandler_(args...); }

protected:
    void SetEntryHandler(EntryType entryHandler) { entryHandler_ = std::move(entryHandler); }

private:
    EntryType entryHandler_;
};

template <PluginKind tkind, typename TRetType, typename... TArgType>
class SharedLibraryPlugin : public Plugin<tkind, TRetType, TArgType...> {
public:
    SharedLibraryPlugin(std::string name, std::string libName, std::string libVersion, std::string entryName)
        : Plugin<tkind, TRetType, TArgType...>(std::move(name), EntryType{}),
          library_(std::move(libName), std::move(libVersion)),
          entryName_(std::move(entryName))
    {
        Load();
    }

    SharedLibraryPlugin(const SharedLibraryPlugin&) = delete;
    SharedLibraryPlugin& operator=(const SharedLibraryPlugin&) = delete;

    ~SharedLibraryPlugin() override { Unload(); }

    bool IsValid() const { return library_.Valid(); }

    bool Load() override
    {
        if (!library_.Load()) {
            this->SetEntryHandler(EntryType{});
            return false;
        }
        auto entry = reinterpret_cast<Entry>(library_.GetSymbolAddress(entryName_));
        if (entry == nullptr) {
            library_.Unload();
            this->SetEntryHandler(EntryType{});
            return false;
        }
        this->SetEntryHandler(entry);
        return true;
    }

    void Unload() override
    {
        this->SetEntryHandler(EntryType{});
        library_.Unload();
    }

private:
    using Entry = TRetType (*)(TArgType...);
    using EntryType = typename Plugin<tkind, TRetType, TArgType...>::EntryType;

    SharedLibrary library_;
    std::string entryName_;
};

class PluginManager {
public:
    static PluginManager& GetInstance();

    void ClearPlugin();

    template <typename TPlugin>
    std::vector<std::shared_ptr<TPlugin>> GetPlugin()
    {
        std::vector<std::shared_ptr<PluginBase>>& pluginBaseList = pluginListDict_[TPlugin::kind];

        std::vector<std::shared_ptr<TPlugin>> pluginList;
        for (auto& pluginBase : pluginBaseList) {
            auto plugin = std::static_pointer_cast<TPlugin>(pluginBase);
            pluginList.push_back(plugin);
        }
        return pluginList;
    }

    template <typename TPlugin>
    bool AddPlugin(const std::string& name, typename TPlugin::EntryType entryHandler)
    {
        auto plugin = std::make_shared<TPlugin>(name, std::move(entryHandler));
        return AddPlugin(std::static_pointer_cast<PluginBase>(plugin));
    }

    template <typename TPlugin>
    bool AddSharedLibraryPlugin(const std::string& name, const std::string& libName, const std::string& libVersion,
                                const std::string& entryName)
    {
        auto plugin = std::make_shared<TPlugin>(name, libName, libVersion, entryName);
        if (!plugin->IsValid()) {
            return false;
        }
        return AddPlugin(std::static_pointer_cast<PluginBase>(plugin));
    }

    template <typename TPlugin, typename... TArgs>
    std::vector<typename TPlugin::RetType> RunPlugin(TArgs&&... args)
    {
        auto pluginList = GetPlugin<TPlugin>();
        std::vector<typename TPlugin::RetType> results;
        results.reserve(pluginList.size());
        for (auto& plugin : pluginList) {
            results.push_back(plugin->Call(args...));
        }
        return results;
    }

private:
    bool AddPlugin(const std::shared_ptr<PluginBase>& plugin);

    std::unordered_map<PluginKind, std::vector<std::shared_ptr<PluginBase>>> pluginListDict_;
    std::unordered_map<std::string, std::shared_ptr<PluginBase>> pluginDict_;
};

using PluginCodegenSrc = Plugin<PluginKind::CODEGEN_SRC, int, const std::string&, std::string&>;

using PluginPassPGOVF = SharedLibraryPlugin<PluginKind::PASS_PGO_VF, Status, const pypto::ir::ProgramPtr&,
                                            const pypto::ir::FunctionPtr&, pypto::ir::ProgramPtr&,
                                            pypto::ir::FunctionPtr&>;

} // namespace npu::tile_fwk
