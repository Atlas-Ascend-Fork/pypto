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
 * \file kernel_bundle_loader.cpp
 * \brief Implementation of the kernel bundle loader.
 */

#include "machine/runtime/bundle/kernel_bundle_loader.h"

#include <iomanip>
#include <sstream>

#include "machine/runtime/bundle/kernel_bundle_crc32.h"
#include "machine/runtime/bundle/kernel_bundle_format.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "utils/file_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::bundle {

namespace {
// Copy one TLV value out of the (possibly transient) source image into a buffer the LoadedBundle owns.
//
// Opaque device segments need no additional host alignment. CtrlFlowCache is also rebased
// and traversed on the host; copy it separately into CtrlFlowCacheBlob so stitch nodes
// retain the 64-byte alignment required by their tagged next pointers.
std::vector<uint8_t> CopySegment(const uint8_t* src, size_t len)
{
    std::vector<uint8_t> v(len);
    if (len != 0) {
        MemcpyS(v.data(), v.size(), src, len);
    }
    return v;
}

// Fold one TLV's identity (type, content CRC, length) into a running 64-bit FNV-1a digest. The result
// identifies the bundle by CONTENT, which is what the registry keys on -- see the bundleKey comment in
// kernel_bundle_loader.h for why hashKey cannot serve that role.
void FoldTlvIntoBundleKey(uint64_t& key, uint32_t type, uint32_t crc, uint64_t len)
{
    constexpr uint64_t kFnvPrime = 0x00000100000001B3ULL;
    for (uint64_t word : {static_cast<uint64_t>(type), static_cast<uint64_t>(crc), len}) {
        for (int byte = 0; byte < 8; ++byte) {
            key = (key ^ ((word >> (byte * 8)) & 0xFFULL)) * kFnvPrime;
        }
    }
}

std::string BundleKeyName(uint64_t bundleKey)
{
    std::ostringstream os;
    os << "PyPTO_bundle_" << std::hex << std::setw(16) << std::setfill('0') << bundleKey;
    return os.str();
}

void ResetCachedOperandUseList(dynamic::DevAscendFunction& func,
                               dynamic::DevLocalVector<dynamic::DevAscendFunctionCallOperandUse>& uses)
{
    for (size_t useIdx = 0; useIdx < uses.size(); ++useIdx) {
        func.At(uses, static_cast<int>(useIdx)).cachedAttrBase = nullptr;
    }
}

// Host control-flow execution can populate these lazy caches before an exception bundle is
// dumped. They are process-local pointers, not relocatable program data. Clear all operand
// use lists so the consumer recomputes each pointer from its own function and offsetAttrIdx.
void ResetCachedOperandAttributes(std::vector<uint8_t>& program)
{
    auto* prog = reinterpret_cast<dynamic::DevAscendProgram*>(program.data());
    const auto base = reinterpret_cast<uint64_t>(prog);
    prog->RelocProgram(0, base);
    for (size_t i = 0; i < prog->GetFunctionSize(); ++i) {
        auto* func = prog->GetFunction(static_cast<int>(i));
        for (size_t incastIdx = 0; incastIdx < func->GetIncastSize(); ++incastIdx) {
            auto& incast = func->GetIncast(static_cast<int>(incastIdx));
            ResetCachedOperandUseList(*func, incast.consumerList);
            ResetCachedOperandUseList(*func, incast.stitchPolicyFullCoverConsumerList);
        }
        for (size_t outcastIdx = 0; outcastIdx < func->GetOutcastSize(); ++outcastIdx) {
            auto& outcast = func->GetOutcast(static_cast<int>(outcastIdx));
            ResetCachedOperandUseList(*func, outcast.producerConsumerList);
            ResetCachedOperandUseList(*func, outcast.stitchPolicyFullCoverProducerList);
        }
    }
    prog->RelocProgram(base, 0);
}

std::string BundlePathName(const std::string& path, uint64_t bundleKey)
{
    const size_t separator = path.find_last_of("/\\");
    std::string name = separator == std::string::npos ? path : path.substr(separator + 1);
    constexpr const char* suffix = ".pyptokb";
    constexpr size_t suffixSize = 8;
    if (name.size() >= suffixSize && name.compare(name.size() - suffixSize, suffixSize, suffix) == 0) {
        name.erase(name.size() - suffixSize);
    }
    if (name.empty()) {
        return BundleKeyName(bundleKey);
    }
    if (name.compare(0, 5, "PyPTO") != 0) {
        name = "PyPTO_bundle_" + name;
    }
    return name;
}
} // namespace

std::shared_ptr<LoadedBundle> KernelBundleLoader::LoadFromMemory(const uint8_t* p, size_t n)
{
    if (p == nullptr || n < sizeof(BundleHeader)) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] buffer too small: %zu", n);
        return nullptr;
    }
    BundleHeader hdr{};
    MemcpyS(&hdr, sizeof(hdr), p, sizeof(BundleHeader));
    if (hdr.magic != kBundleMagic) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] bad magic %#lx", hdr.magic);
        return nullptr;
    }
    if (hdr.version > kBundleVersion) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] unsupported version %u > %u", hdr.version,
                     kBundleVersion);
        return nullptr;
    }
    if (hdr.totalSize != n) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] size mismatch: header=%lu bytes actual=%zu bytes",
                     hdr.totalSize, n);
        return nullptr;
    }
    if (Crc32(p, sizeof(BundleHeader) - sizeof(uint32_t)) != hdr.headerCrc32) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] header CRC mismatch");
        return nullptr;
    }
    const uint64_t tlvRegion = sizeof(BundleHeader) + static_cast<uint64_t>(hdr.tlvCount) * sizeof(TlvHeader);
    if (tlvRegion > n) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] TLV table out of range");
        return nullptr;
    }

    auto b = std::make_shared<LoadedBundle>();
    uint64_t bundleKey = 0xCBF29CE484222325ULL; // FNV-1a offset basis
    for (uint32_t i = 0; i < hdr.tlvCount; ++i) {
        TlvHeader t{};
        MemcpyS(&t, sizeof(t), p + sizeof(BundleHeader) + static_cast<size_t>(i) * sizeof(TlvHeader),
                sizeof(TlvHeader));
        if (t.valueOffset > n || t.valueLength > n - t.valueOffset) {
            MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] TLV[%u] value out of range", i);
            return nullptr;
        }
        const uint8_t* vp = p + t.valueOffset;
        if (Crc32(vp, t.valueLength) != t.valueCrc32) {
            MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] TLV[%u] value CRC mismatch", i);
            return nullptr;
        }
        FoldTlvIntoBundleKey(bundleKey, t.type, t.valueCrc32, t.valueLength);
        switch (static_cast<TlvType>(t.type)) {
            case TlvType::AicoreKernel:
                b->aicoreKernel = CopySegment(vp, t.valueLength);
                break;
            case TlvType::AicpuSo:
                b->aicpuSo = CopySegment(vp, t.valueLength);
                break;
            case TlvType::DevProgram:
                b->devProgram = CopySegment(vp, t.valueLength);
                break;
            case TlvType::CtrlFlowCache:
                b->ctrlFlowCache.assign(vp, vp + t.valueLength);
                break;
            case TlvType::SymbolMeta:
                b->symbolMeta = CopySegment(vp, t.valueLength);
                break;
            default:
                MACHINE_LOGI("[kernel-bundle] skip unknown TLV type %u", t.type);
                break;
        }
    }
    if (b->devProgram.empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] missing DEV_PROGRAM segment");
        return nullptr;
    }
    ResetCachedOperandAttributes(b->devProgram);
    b->bundleKey = bundleKey;
    b->displayName = BundleKeyName(bundleKey);
    MACHINE_LOGI("[kernel-bundle] loaded: bundleKey=%#lx hashKey=%#lx workspaceSize=%lu archInfo=%u ctrlCache=%zuB "
                 "symbolMeta=%zuB",
                 b->bundleKey, b->GetHashKey(), b->GetWorkspaceSize(), b->GetArchInfo(), b->ctrlFlowCache.size(),
                 b->symbolMeta.size());
    return b;
}

std::shared_ptr<LoadedBundle> KernelBundleLoader::LoadFromFile(const std::string& path)
{
    bool ok = false;
    std::vector<uint8_t> raw = ReadFile(path, &ok);
    if (!ok || raw.empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "[kernel-bundle] failed to read %s", path.c_str());
        return nullptr;
    }
    auto bundle = LoadFromMemory(raw.data(), raw.size());
    if (bundle != nullptr) {
        bundle->displayName = BundlePathName(path, bundle->bundleKey);
    }
    return bundle;
}

} // namespace npu::tile_fwk::bundle
