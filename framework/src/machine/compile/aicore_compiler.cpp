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
 * \file aicore_compiler.cpp
 * \brief
 */

#include "machine/compile/aicore_compiler.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "codegen/utils/parallel_execute.h"
#include "utils/file_utils.h"
#include "interface/utils/common.h"
#include "interface/utils/op_info_manager.h"
#include "machine/compile/gen_aicore_code.h"
#include "machine/host/main_block.h"
#include "machine/utils/checkinject.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
namespace {
constexpr const size_t CMD_SIZE_1K = 1024;
constexpr const size_t CMD_SIZE_4K = 4096;
constexpr const char* BISHENG_PROGRAM_CMD = "bisheng";
constexpr const char* BISHENG_LD_CMD = "ld.lld";
constexpr const char* RUN_TO_LOG_SCRIPT_NAME = "run_to_log.sh";
static std::string QuoteShellArg(const std::string& arg)
{
    std::string escaped = "'";
    for (char c : arg) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped += c;
        }
    }
    escaped += "'";
    return escaped;
}

static std::string GetRunToLogScriptPath()
{
    const std::string installed = GetPyptoLibPath() + "/scripts/" + RUN_TO_LOG_SCRIPT_NAME;
    if (IsPathExist(installed)) {
        return installed;
    }
    // Source-tree fallback for local UT / uninstalled builds.
    std::string srcDir(__FILE__);
    const auto pos = srcDir.find_last_of('/');
    if (pos == std::string::npos) {
        srcDir = ".";
    } else {
        srcDir.resize(pos);
    }
    const std::string srcPath = srcDir + "/scripts/" + RUN_TO_LOG_SCRIPT_NAME;
    return IsPathExist(srcPath) ? srcPath : "";
}

static void LogBishengCompileFailure(const std::string& logPath)
{
    bool readOk = false;
    const auto logBytes = ReadFile(logPath, &readOk);
    if (readOk && !logBytes.empty()) {
        const std::string log(reinterpret_cast<const char*>(logBytes.data()), logBytes.size());
        MACHINE_LOGE(HostBackEndErr::COMPILE_CCEC_FAILED, "Compile ccec failed.\n%s", log.c_str());
    } else {
        MACHINE_LOGE(HostBackEndErr::COMPILE_CCEC_FAILED, "Compile ccec failed.");
    }
}

static bool IsPtoDebugRoot(const std::string& root)
{
    return IsPathExist(root + "/impl/utils/debug/pto_aicore_printf_impl.h") &&
           IsPathExist(root + "/include/utils/debug/pto_dump.h");
}

// Vendored under tilefwk/pto_debug (installed with device headers). No longer needs $ASCEND_HOME/asc.
static std::string ResolvePtoDebugRoot()
{
    const std::string installed = GetPyptoLibPath() + "/include/tilefwk/pto_debug";
    if (IsPtoDebugRoot(installed)) {
        return installed;
    }
    // Source-tree fallback: .../machine/compile -> .../interface/machine/device/tilefwk/pto_debug
    std::string srcDir(__FILE__);
    const auto pos = srcDir.find_last_of('/');
    if (pos != std::string::npos) {
        srcDir.resize(pos);
    }
    const std::string fromSrc = srcDir + "/../../interface/machine/device/tilefwk/pto_debug";
    if (IsPtoDebugRoot(fromSrc)) {
        return fromSrc;
    }
    return "";
}

static std::string BuildPtoPrintfCompileFlags()
{
    if (!IsAicorePrintEnabled()) {
        return "";
    }
    const std::string ptoDebugRoot = ResolvePtoDebugRoot();
    if (ptoDebugRoot.empty()) {
        return "";
    }
    // -I root for "impl/...", -I root/include for "utils/...". Vendored macros omit g_tilingKey.
    return "-D__ENABLE_PTO_PRINTF__ -DPYPTO_PTO_DEBUG=1 -I" + QuoteShellArg(ptoDebugRoot) + " -I" +
           QuoteShellArg(ptoDebugRoot + "/include") + " ";
}

static std::string BuildDrcoOptCompileFlags(bool graphHasMixWork)
{
    return std::string("-DDRCO_FLAG_HAS_MIX=") + (graphHasMixWork ? "1" : "0");
}
} // namespace

// Redirect cannot appear in Checkinject()'d system() args; use packaged/source-tree run_to_log.sh.
int RunBishengQuiet(const std::string& compileCmd, const std::string& workDir)
{
    const std::string dirPrefix = (workDir.empty() || workDir.back() == '/') ? workDir : workDir + "/";
    std::string logTemplate = dirPrefix + "bisheng_log_" + std::to_string(getpid()) + "_XXXXXX";
    std::vector<char> logBuf(logTemplate.begin(), logTemplate.end());
    logBuf.push_back('\0');
    int logFd = mkstemp(logBuf.data());
    if (logFd < 0) {
        MACHINE_LOGE(HostBackEndErr::COMPILE_CCEC_FAILED, "Create bisheng log failed.");
        return -1;
    }
    (void)close(logFd);
    const std::string logPath(logBuf.data());

    const std::string scriptPath = GetRunToLogScriptPath();
    if (scriptPath.empty()) {
        (void)unlink(logPath.c_str());
        MACHINE_LOGE(HostBackEndErr::COMPILE_CCEC_FAILED,
                     "run_to_log.sh not found under lib/scripts or source-tree compile/scripts.");
        return -1;
    }

    const std::string runCmd = "bash " + QuoteShellArg(scriptPath) + " " + QuoteShellArg(logPath) + " " +
                               QuoteShellArg(compileCmd);
    int ret = Checkinject(runCmd.c_str(), runCmd.size());
    if (ret != 0) {
        (void)unlink(logPath.c_str());
        MACHINE_LOGE(HostBackEndErr::COMPILE_CCEC_FAILED, "Bisheng compile cmd illegal char.");
        return ret;
    }
    ret = std::system(runCmd.c_str());
    if (ret != 0) {
        LogBishengCompileFailure(logPath);
    }
    (void)unlink(logPath.c_str());
    return ret;
}

static int CompileCoreMachine(const std::string& objFile, bool isCube, uint64_t tilingKey, const std::string& headFile,
                              const std::string& aicoreSrcFile, const std::string& funcRawName, bool requiresSimt,
                              bool graphHasMixWork, bool l2Close)
{
    MACHINE_LOGI("Compile src file is [%s], kernel type[%d].", aicoreSrcFile.c_str(), isCube);
    const std::string cc_opt = isCube ? Platform::Instance().GetSoc().GetCCECVersion("AIC") :
                                        Platform::Instance().GetSoc().GetCCECVersion("AIV");
    const std::string coreType = isCube ? "-D__AIC__" : "-D__AIV__";
    const auto& opType = OpInfoManager::GetInstance().GetOpType();
    std::string hasSubFunc = headFile.empty() ? "" : "-D__HAS_SUB_FUNC__";
    const std::string davArch = (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201) ? "-D__DAV_V220" :
                                                                                                    "-D__DAV_V310";
    const std::string enableMainBlock = ((config::GetPassGlobalConfig(KEY_ENABLE_VF, false) ||
                                          config::GetRuntimeOption<int64_t>(CFG_VALID_SHAPE_OPTIMIZE) == 1)) ?
                                            "-D__ENABLE_MAIN_BLOCK" :
                                            "";
    const std::string enableAicoreResolve = IsAicoreResolveEnabled() ? "-DENABLE_AICORE_RESOLVE=1" :
                                                                       "-DENABLE_AICORE_RESOLVE=0";
    const std::string enableAicoreBnParallel = (IsAicoreResolveEnabled() && config::GetDeviceSchedParallelism() > 1) ?
                                                   "-DENABLE_AICORE_BN_PARALLEL=1" :
                                                   "-DENABLE_AICORE_BN_PARALLEL=0";
    // DRCO L2 亲和调度编译期开关：device_sched_mode 的 L2CACHE_AFFINITY_SCH 位随调度模式传入，
    // 宏为 0 时设备侧 DRCO 调度代码与非 L2 亲和版本完全一致。
    // 与 并行互斥（临时，待专项融合）：两者同时启用时 AICPU 派发时序与 并行窗口
    // admit/handover 协议存在数据陈旧窗口——并行优先，并行开启时强制关闭 L2 亲和
    const bool bnParallelOn = enableAicoreBnParallel.find("=1") != std::string::npos;
    const std::string drcoL2Close = (l2Close && !bnParallelOn) ? "-DDRCO_FLAG_L2_CLOSE=1" : "-DDRCO_FLAG_L2_CLOSE=0";
    const std::string requiresSimtMacro = requiresSimt ? "-DREQUIRES_SIMT=1" : "-DREQUIRES_SIMT=0";
    const std::string simtCompileOption = requiresSimt ? "-mllvm -cce-dyn-kernel-stack-size=false" : "";
    std::string ccecCmd;
    ccecCmd.resize(CMD_SIZE_4K);
    std::string includePath = GetPyptoLibPath() + "/../include/tile_fwk";
    const std::string ptoPrintfFlags = BuildPtoPrintfCompileFlags();
    const std::string drcoOptFlags = BuildDrcoOptCompileFlags(graphHasMixWork);
    constexpr const char* warningFlags = "-Wall -Werror";
    int ret = snprintf_s(
        ccecCmd.data(), CMD_SIZE_4K, CMD_SIZE_4K - 1,
        "%s -c -O3 -g -x cce %s -std=c++17 "
        "--cce-aicore-only "
        "--cce-aicore-arch=%s "
        "-mllvm -cce-aicore-stack-size=0x8000 "
        "-mllvm -cce-aicore-function-stack-size=0x8000 "
        "%s "
        "-mllvm -cce-aicore-record-overflow=false "
        "-mllvm -cce-aicore-addr-transform "
        "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
        "-D__MIX__ "
        "-D__TILINGKEY__=%s "
        "-D__OPTYPE__=%s "
        "-D__OPNAME__=%s "
        "-D__HEAD_FILE__=%s "
        "%s "
        "%s "
        "%s "
        "%s "
        "%s "
        "%s "
        "%s "
        "%s "
        "-I%s/tileop/arch32 "
        "-I%s/ "
        "-I%s/include/tileop/arch32 "
        "-I%s/include/ "
        "-o %s %s %s %s",
        BISHENG_PROGRAM_CMD, warningFlags, cc_opt.c_str(), simtCompileOption.c_str(), std::to_string(tilingKey).c_str(),
        opType.c_str(), funcRawName.c_str(), headFile.c_str(), requiresSimtMacro.c_str(), hasSubFunc.c_str(),
        coreType.c_str(), enableAicoreResolve.c_str(), enableAicoreBnParallel.c_str(), ptoPrintfFlags.c_str(),
        drcoOptFlags.c_str(), drcoL2Close.c_str(), includePath.c_str(), includePath.c_str(), GetPyptoLibPath().c_str(),
        GetPyptoLibPath().c_str(), objFile.c_str(), aicoreSrcFile.c_str(), davArch.c_str(), enableMainBlock.c_str());
    if (ret < 0) {
        MACHINE_LOGE(HostBackEndErr::COMPILE_AICORE_FAILED, "Compile aicore construct cmd failed.");
        return ret;
    }
    ret = Checkinject(ccecCmd.c_str(), ccecCmd.size());
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::COMPILE_AICORE_FAILED, "Compile aicore cmd illegal char.");
        return ret;
    }
    const auto slash = objFile.find_last_of('/');
    const std::string workDir = (slash == std::string::npos) ? "" : objFile.substr(0, slash + 1);
    return RunBishengQuiet(std::string(ccecCmd.c_str()), workDir);
}

std::string GenSubFuncCall(std::map<uint64_t, Function*>& leafDict, CoreType coreType,
                           dynamic::EncodeDevAscendFunctionParam& param, const std::string& ccePath, uint64_t tilingKey,
                           std::stringstream& src_obj, bool enableSubFunc)
{
    if (!enableSubFunc) {
        return "";
    }
    std::stringstream code;
    std::map<int, std::string> idxNameMap;
    // Declare extern leaf func.
    for (const auto& iter : leafDict) {
        const auto leaf = iter.second;
        int leafIndex = param.calleeHashIndexDict[leaf->ComputeHash().GetHash()];
        auto leafFuncAttr = leaf->GetLeafFuncAttribute();
        ASSERT(ProgEncodeErr::LEAF_CALLEE_ATTR_NULL, leafFuncAttr != nullptr) << "LeafFuncAttr is null for " << leaf;
        if (coreType != leafFuncAttr->coreType) {
            continue;
        }
        int baseIdx = leafFuncAttr->binPathMainBlock.empty() ? leafIndex : leafIndex * MAIN_BLOCK_SIZE - 1;
        src_obj << leafFuncAttr->binPath << " ";
        code << leafFuncAttr->kernelDeclare << std::endl;
        idxNameMap[baseIdx] = leafFuncAttr->kernelName;
        if (!leafFuncAttr->binPathMainBlock.empty()) {
            src_obj << leafFuncAttr->binPathMainBlock << " ";
            code << leafFuncAttr->kernelDeclareMainBlock << std::endl;
            idxNameMap[baseIdx + 1] = leafFuncAttr->kernelNameMainBlock;
        }
        MACHINE_LOGD("Func[%d], kernel_name[%s].", leafIndex, leafFuncAttr->kernelName.c_str());
    }
    if (idxNameMap.empty()) {
        return "";
    }
    // Define call sub func inline function.
    code << "__attribute__((always_inline)) inline __aicore__ void CallSubFuncTask(uint64_t funcIdx, ";
    code << "CoreFuncParam *param, int64_t gmStackAddr, __gm__ int64_t *hcclContext, __gm__ FedProfLeafTaskProf* "
            "taskStat) {\n";
    code << "    switch (funcIdx) {\n";
    for (const auto& iter : idxNameMap) {
        MACHINE_LOGD("Call sub func id[%d], kernel_name[%s].", iter.first, iter.second.c_str());
        code << "        case " << std::to_string(iter.first) << ": {\n";
        code << "            " << iter.second << "(param, gmStackAddr, hcclContext, taskStat);\n";
        code << "            break;\n";
        code << "        }\n";
    }
    code << "        default:\n";
    code << "            return;\n";
    code << "    };\n    return;\n}\n";
    MACHINE_LOGD("Sub func call code [\n%s\n].", code.str().c_str());

    std::string head_file = (coreType == CoreType::AIC) ? "sub_func_aic_call_" : "sub_func_aiv_call_";
    head_file = ccePath + head_file + std::to_string(tilingKey) + ".h";
    FILE* fsrc = fopen(head_file.c_str(), "w");
    if (fsrc == nullptr) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "Fail to open call.h.");
        return "";
    }
    (void)fprintf(fsrc, "%s", code.str().c_str());
    (void)fclose(fsrc);
    return head_file;
}

static int RunLdViaSecureTempScript(const std::string& ccePath, const std::string& key, const std::string& ccecCmd)
{
    std::string linkScriptTemplate = ccePath + "link_" + key + std::to_string(getpid()) + "_XXXXXX";
    std::vector<char> linkScriptBuffer(linkScriptTemplate.begin(), linkScriptTemplate.end());
    linkScriptBuffer.push_back('\0');
    int fd = mkstemp(linkScriptBuffer.data());
    if (fd < 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Create secure link script failed.");
        return -1;
    }
    const std::string linkScript(linkScriptBuffer.data());
    if (fchmod(fd, S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Set secure link script permission failed, file[%s].",
                     linkScript.c_str());
        (void)close(fd);
        (void)unlink(linkScript.c_str());
        return -1;
    }
    const std::string scriptContent = std::string("#!/bin/bash\n") + ccecCmd;
    ssize_t writtenSize = write(fd, scriptContent.data(), scriptContent.size());
    if (writtenSize < 0 || static_cast<size_t>(writtenSize) != scriptContent.size()) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Write secure link script failed, file[%s].", linkScript.c_str());
        (void)close(fd);
        (void)unlink(linkScript.c_str());
        return -1;
    }
    if (close(fd) != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Close secure link script failed, file[%s].", linkScript.c_str());
        (void)unlink(linkScript.c_str());
        return -1;
    }
    const std::string ldCmd = "bash " + linkScript;
    int ret = Checkinject(ldCmd.c_str(), ldCmd.size());
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Link kernel cmd illegal char.");
        return ret;
    }
    ret = std::system(ldCmd.c_str());
    if (unlink(linkScript.c_str()) != 0) {
        MACHINE_LOGW("Cleanup secure link script failed, file[%s].", linkScript.c_str());
    }
    if (ret != 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "Link kernel failed.");
    }
    return ret;
}

static int LinkObject(const std::string& src_objs, std::string& objPath, const std::string& ccePath, bool relocate,
                      const std::string& key)
{
    size_t cmdSize = src_objs.size() + objPath.size() + CMD_SIZE_1K;
    std::string ccecCmd;
    ccecCmd.resize(cmdSize);
    int ret = snprintf_s(ccecCmd.data(), cmdSize, cmdSize - 1,
                         "%s -m aicorelinux -Ttext=0 -static %s -o "
                         "%s "
                         "%s",
                         BISHENG_LD_CMD, relocate ? "-r" : "", objPath.c_str(), src_objs.c_str());
    if (ret < 0) {
        MACHINE_LOGE(HostBackEndErr::LINK_FAILED, "LinkCoreMachine construct cmd failed.");
        return ret;
    }
    MACHINE_LOGD("Link ccec command: [%s].", ccecCmd.c_str());
    return RunLdViaSecureTempScript(ccePath, key, ccecCmd);
}

bool DynamicKernelRequiresSimt(const std::map<uint64_t, Function*>& leafDict)
{
    const std::string simtAttr = OP_ATTR_PREFIX + "requires_simt";
    for (const auto& [hash, leaf] : leafDict) {
        (void)hash;
        for (Operation& op : leaf->Operations()) {
            if (op.HasAttribute(simtAttr) && op.GetBoolAttribute(simtAttr)) {
                return true;
            }
        }
    }
    return false;
}

int CompileAICoreKernel(std::map<uint64_t, Function*>& leafDict, dynamic::EncodeDevAscendFunctionParam& param,
                        const std::string& ccePath, const std::string& funcHash, const std::string& funcRawName,
                        std::string& kernelPath, bool requiresSimt, bool graphHasMixWork, bool enableSubFunc,
                        bool l2Close)
{
    if (ccePath.empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "No cce path.");
        return -1;
    }
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    std::string aic_obj = ccePath + "dy_kernel_" + funcHash + "_aic_" + std::to_string(tilingKey) + ".o";
    std::string aiv_obj = ccePath + "dy_kernel_" + funcHash + "_aiv_" + std::to_string(tilingKey) + ".o";
    std::string aicoreSrcFile = ccePath + "aicore_" + funcHash + ".cpp";
    if (!GenAicoreSrcFile(aicoreSrcFile)) {
        MACHINE_LOGE(HostBackEndErr::GEN_AICORE_FILE_FAILED, "Fail to generate aicore src file.");
        return -1;
    }
    std::deque<std::function<void(void)>> tasks;
    auto task = [&ccePath, &funcHash, &funcRawName, &leafDict, &param, &aic_obj, &aicoreSrcFile, &tilingKey,
                 enableSubFunc, requiresSimt, graphHasMixWork, l2Close]() {
        // gen switch case func
        std::stringstream src_aic_obj;
        auto headFile = GenSubFuncCall(leafDict, CoreType::AIC, param, ccePath, tilingKey, src_aic_obj, enableSubFunc);
        std::string mid_aic_obj = ccePath + "mid_kernel_" + funcHash + "_aic_" + std::to_string(tilingKey) + ".o";
        auto ret = CompileCoreMachine(mid_aic_obj, true, tilingKey, headFile, aicoreSrcFile, funcRawName, requiresSimt,
                                      graphHasMixWork, l2Close);
        ASSERT(HostBackEndErr::COMPILE_AICORE_FAILED, ret == 0)
            << "CompileCoreMachine failed with return code  " << ret;
        src_aic_obj << mid_aic_obj;
        ret = LinkObject(src_aic_obj.str(), aic_obj, ccePath, true, "aic");
        ASSERT(HostBackEndErr::LINK_FAILED, ret == 0) << "LinkObject failed with return code  " << ret;
        return;
    };
    tasks.push_back(task);

    auto task1 = [&ccePath, &funcHash, &funcRawName, &leafDict, &param, &aiv_obj, &aicoreSrcFile, &tilingKey,
                  enableSubFunc, requiresSimt, graphHasMixWork, l2Close]() {
        std::stringstream src_aiv_obj;
        auto headFile = GenSubFuncCall(leafDict, CoreType::AIV, param, ccePath, tilingKey, src_aiv_obj, enableSubFunc);
        std::string mid_aiv_obj = ccePath + "mid_kernel_" + funcHash + "_aiv_" + std::to_string(tilingKey) + ".o";
        auto ret = CompileCoreMachine(mid_aiv_obj, false, tilingKey, headFile, aicoreSrcFile, funcRawName, requiresSimt,
                                      graphHasMixWork, l2Close);
        ASSERT(HostBackEndErr::COMPILE_AICORE_FAILED, ret == 0)
            << "CompileCoreMachine failed with return code  " << ret;
        src_aiv_obj << mid_aiv_obj;
        ret = LinkObject(src_aiv_obj.str(), aiv_obj, ccePath, true, "aiv");
        ASSERT(HostBackEndErr::LINK_FAILED, ret == 0) << "LinkObject failed with return code  " << ret;
        return;
    };
    tasks.push_back(task1);
    ParallelExecuteAndWait(tasks.size(), tasks);

    std::stringstream src_obj;
    src_obj << " " << aic_obj << " " << aiv_obj;
    kernelPath = ccePath + "dy_kernel_" + funcHash + "_" + std::to_string(tilingKey) + ".o";
    MACHINE_LOGD("Compile dynamic kernel to %s.", kernelPath.c_str());
    auto ret = LinkObject(src_obj.str(), kernelPath, ccePath, false, "mix");
    return ret;
}

} // namespace npu::tile_fwk
