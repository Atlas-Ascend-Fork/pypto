/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cce_split.h"
#include "tilefwk/error.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace npu::tile_fwk {
namespace {
bool CountsTowardSplitThreshold(const CceSplitRegion& region)
{
    return region.effect == CceSplitEffect::NON_SYNC_OP || region.effect == CceSplitEffect::EXTERNAL_STATE;
}

std::string Declaration(const std::string& name, const std::string& parameters, bool isPart)
{
    // Runtime must not register internal parts as kernel entry points.
    const std::string visibility = isPart ? "__attribute__((visibility(\"hidden\"))) " : "";
    const std::string attribute = isPart ? "__aicore__" : "[aicore]";
    return "extern \"C\" " + attribute + " " + visibility + "void " + name + parameters;
}

// Scan identifiers only against names supplied by codegen. This does not infer
// declaration names or cut C++ statements. Comments and literals are not uses.
std::vector<size_t> PreambleReferences(const std::string& code,
                                       const std::unordered_map<std::string_view, size_t>& names)
{
    std::vector<size_t> references;
    const std::string_view text(code);
    size_t pos = 0;
    while (pos < text.size()) {
        if (text.substr(pos, 2) == "//") {
            const auto end = text.find('\n', pos + 2);
            pos = end == text.npos ? text.size() : end + 1;
        } else if (text.substr(pos, 2) == "/*") {
            const auto end = text.find("*/", pos + 2);
            pos = end == text.npos ? text.size() : end + 2;
        } else if (text.substr(pos, 2) == "R\"" || text.substr(pos, 3) == "LR\"" || text.substr(pos, 3) == "uR\"" ||
                   text.substr(pos, 3) == "UR\"" || text.substr(pos, 4) == "u8R\"") {
            const auto delimiter = text.find('"', pos) + 1;
            const auto open = text.find('(', delimiter);
            if (open == text.npos) {
                break;
            }
            const std::string close = ")" + std::string(text.substr(delimiter, open - delimiter)) + "\"";
            const auto end = text.find(close, open + 1);
            pos = end == text.npos ? text.size() : end + close.size();
        } else if (text[pos] == '"' || text[pos] == '\'') {
            const char quote = text[pos++];
            while (pos < text.size()) {
                const char c = text[pos++];
                if (c == '\\' && pos < text.size()) {
                    ++pos;
                } else if (c == quote) {
                    break;
                }
            }
        } else if (std::isalpha(static_cast<unsigned char>(text[pos])) || text[pos] == '_') {
            const auto begin = pos++;
            while (pos < text.size() && (std::isalnum(static_cast<unsigned char>(text[pos])) || text[pos] == '_')) {
                ++pos;
            }
            const auto it = names.find(text.substr(begin, pos - begin));
            if (it != names.end()) {
                references.push_back(it->second);
            }
        } else {
            ++pos;
        }
    }
    return references;
}

struct PreambleDependencies {
    std::vector<std::vector<size_t>> definitions;
    std::vector<std::vector<size_t>> regions;

    bool Build(const CceOriginalFunctionBody& body)
    {
        std::unordered_map<std::string_view, size_t> names;
        names.reserve(body.preambleDefinitions.size());
        definitions.reserve(body.preambleDefinitions.size());
        regions.reserve(body.regions.size());
        for (size_t i = 0; i < body.preambleDefinitions.size(); ++i) {
            if (!names.emplace(body.preambleDefinitions[i].first, i).second) {
                return false;
            }
        }
        for (const auto& definition : body.preambleDefinitions) {
            definitions.push_back(PreambleReferences(definition.second, names));
        }
        for (const auto& region : body.regions) {
            auto references = PreambleReferences(region.code, names);
            for (const auto& declaration : region.declarations) {
                auto uses = PreambleReferences(declaration.second, names);
                references.insert(references.end(), uses.begin(), uses.end());
            }
            regions.push_back(std::move(references));
        }
        return true;
    }

    size_t Include(size_t region, const PreambleDefinitions& preamble, std::unordered_set<size_t>& selected,
                   std::vector<size_t>& order) const
    {
        size_t bytes = 0;
        std::vector<size_t> pending = regions[region];
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (!selected.insert(index).second) {
                continue;
            }
            order.push_back(index);
            bytes += preamble[index].second.size();
            pending.insert(pending.end(), definitions[index].begin(), definitions[index].end());
        }
        return bytes;
    }
};

bool FindBoundaries(const CceOriginalFunctionBody& body, std::vector<bool>& boundaries)
{
    std::map<std::string, std::string> definitions;
    bool hasBodyOp = false;
    bool externalStateActive = false;
    for (size_t index = 0; index < body.regions.size(); ++index) {
        const auto& region = body.regions[index];
        for (const auto& [name, code] : region.declarations) {
            auto [it, inserted] = definitions.emplace(name, code);
            if (!inserted && it->second != code) {
                return false;
            }
        }
        switch (region.effect) {
            case CceSplitEffect::NON_SYNC_OP:
            case CceSplitEffect::SET:
            case CceSplitEffect::WAIT:
            case CceSplitEffect::CROSS_CORE_SYNC:
                // Preserve synchronization instructions in source order. Their matching
                // endpoint may be in another segment, leaf or core; do not require local pairing.
                hasBodyOp = true;
                break;
            case CceSplitEffect::BARRIER:
            case CceSplitEffect::FULL_BARRIER:
                break;
            case CceSplitEffect::EXTERNAL_STATE:
                // Local PIPE_ALL cannot establish completion of remote work. Keep this
                // operation and the remaining body together, while still allowing prefix splits.
                externalStateActive = true;
                hasBodyOp = true;
                break;
            default:
                throw std::invalid_argument("Invalid CCE split effect");
        }
        boundaries.push_back(region.allowCutAfter && !externalStateActive && hasBodyOp);
        if (boundaries.back()) {
            hasBodyOp = false;
        }
    }
    return true;
}
} // namespace

CceSplitPlan PlanCceSplit(const CceOriginalFunctionBody& body, const std::string& includes,
                          const std::string& kernelName, const std::string& parameters, const std::string& headerName,
                          int64_t threshold)
{
    if (threshold < 0) {
        throw std::invalid_argument("cce_split_threshold_tile_ops must be nonnegative");
    }
    CceSplitPlan plan;
    const auto operationCount = std::count_if(body.regions.begin(), body.regions.end(), CountsTowardSplitThreshold);
    if (threshold == 0 || static_cast<uint64_t>(operationCount) <= static_cast<uint64_t>(threshold)) {
        return plan;
    }
    if (!body.supported) {
        return plan;
    }
    std::vector<bool> boundaries;
    if (!FindBoundaries(body, boundaries)) {
        return plan;
    }
    PreambleDependencies dependencies;
    if (!dependencies.Build(body)) {
        return plan;
    }
    // Keep the synchronization-only suffix with the last non-synchronization operation.
    const auto lastOperation = std::find_if(body.regions.rbegin(), body.regions.rend(), CountsTowardSplitThreshold);
    const auto lastOperationEnd = static_cast<size_t>(std::distance(body.regions.begin(), lastOperation.base()));
    const auto partNumberWidth = std::to_string(operationCount).size();
    size_t start = 0;
    const std::string prefix = includes + "#include \"" + headerName + "\"\n";
    while (start < body.regions.size()) {
        CceGeneratedSegment part;
        const auto partNumber = std::to_string(plan.parts.size());
        part.name = kernelName + "_part" + std::string(partNumberWidth - partNumber.size(), '0') + partNumber;
        const std::string wrapper = prefix + Declaration(part.name, parameters, true) + " {\n";
        std::string source;
        std::unordered_set<size_t> selected;
        std::vector<size_t> preambleOrder;
        size_t preambleBytes = 0;
        size_t chosenPreambleCount = 0;
        size_t chosenSourceBytes = 0;
        size_t chosenPreambleBytes = 0;
        std::set<std::string> emitted;
        size_t chosenEnd = start;
        size_t partOperationCount = 0;
        for (size_t i = start; i < body.regions.size(); ++i) {
            const auto& region = body.regions[i];
            partOperationCount += CountsTowardSplitThreshold(region) ? 1 : 0;
            for (const auto& [name, definition] : region.declarations) {
                if (emitted.insert(name).second) {
                    source += definition;
                }
            }
            source += region.code;
            preambleBytes += dependencies.Include(i, body.preambleDefinitions, selected, preambleOrder);
            // Never select a synchronization-only prefix, or cut before a synchronization-only suffix.
            if (partOperationCount == 0 ||
                ((!boundaries[i] || i + 1 >= lastOperationEnd) && i + 1 != body.regions.size())) {
                continue;
            }
            const bool exceeds = partOperationCount > static_cast<uint64_t>(threshold);
            if (exceeds && chosenEnd != start) {
                break;
            }
            chosenEnd = i + 1;
            chosenSourceBytes = source.size();
            chosenPreambleCount = preambleOrder.size();
            chosenPreambleBytes = preambleBytes;
            if (exceeds) {
                break;
            }
        }
        if (chosenEnd == start) {
            plan.parts.clear();
            return plan;
        }
        // Materialize only the chosen candidate; avoid copying each growing prefix.
        preambleOrder.resize(chosenPreambleCount);
        std::sort(preambleOrder.begin(), preambleOrder.end());
        part.source.reserve(wrapper.size() + chosenPreambleBytes + chosenSourceBytes + 2);
        part.source += wrapper;
        for (const auto index : preambleOrder) {
            part.source += body.preambleDefinitions[index].second;
        }
        part.source.append(source, 0, chosenSourceBytes);
        part.source += "}\n";
        plan.parts.push_back(std::move(part));
        start = chosenEnd;
    }
    if (plan.parts.size() < 2) {
        plan.parts.clear();
        return plan;
    }
    plan.header = "#pragma once\n";
    plan.entry = prefix + Declaration(kernelName, parameters, false) + " {\n";
    for (const auto& part : plan.parts) {
        plan.header += Declaration(part.name, parameters, true) + ";\n";
        // Bisheng can drop visibility attributes on aicore functions; enforce ELF visibility as well.
        plan.header += "asm(\".hidden " + part.name + "\");\n";
        // Keep arguments in sync with CodeGenNPU::GenFuncHeader (codegen_npu.cpp) parameter names and order.
        plan.entry += part.name + "(param, GMStackBase, hcclContext, taskStat);\n";
    }
    plan.entry += "}\n";

    return plan;
}

std::string QuoteCceBuildArg(const std::string& value)
{
    ASSERT(CmpCodeErr::CMD_CHECK_FAILED,
           value.find_first_of("\n\r") == std::string::npos && value.find('\0') == std::string::npos)
        << "CCE build paths must not contain newline, carriage return or NUL";
    std::string result = "'";
    for (char ch : value) {
        result += ch == '\'' ? "'\\''" : std::string(1, ch);
    }
    return result + "'";
}

static std::string MakeRecipe(const std::string& command)
{
    std::string result;
    for (char ch : command) {
        if (ch == '$') {
            result += '$';
        }
        result += ch;
    }
    return result;
}

static std::string MakeDependency(const std::string& value, bool target = false)
{
    (void)QuoteCceBuildArg(value);
    ASSERT(CmpCodeErr::CMD_CHECK_FAILED, value.find_first_of(";|") == std::string::npos)
        << "CCE Makefile paths must not contain semicolon or pipe";
    std::string result;
    size_t backslashes = 0;
    for (char ch : value) {
        if (ch == '\\') {
            ++backslashes;
            continue;
        }
        const bool escape = ch == ' ' || ch == ':' || ch == '\t' || ch == '#' || (target && ch == '%');
        result.append(backslashes * (escape ? 2 : 1), '\\');
        backslashes = 0;
        if (escape) {
            result += '\\';
        } else if (ch == '$') {
            result += '$';
        }
        result += ch;
    }
    // A following separator must not be escaped by a trailing path backslash.
    result.append(backslashes * 2, '\\');
    return result;
}

std::string RenderCceMakefile(std::vector<CompileTaskInfo> tasks)
{
    std::ostringstream makefile;
    // 任务按并行完成顺序入列，排序后写出才能让Makefile内容与线程调度无关
    std::sort(tasks.begin(), tasks.end(),
              [](const CompileTaskInfo& lhs, const CompileTaskInfo& rhs) { return lhs.outputPath < rhs.outputPath; });

    makefile << "# Auto-generated Makefile for parallel compilation\n";

    makefile << "all: ";
    for (const auto& task : tasks) {
        makefile << MakeDependency(task.outputPath) << " ";
    }
    makefile << "\n\n";

    for (const auto& task : tasks) {
        makefile << MakeDependency(task.outputPath, true) << ": " << MakeDependency(task.inputPath);
        for (const auto& dependency : task.dependencies) {
            makefile << " " << MakeDependency(dependency);
        }
        makefile << "\n";
        makefile << "\t@" << MakeRecipe(task.compileCmd) << "\n\n";
        if (!task.publishFrom.empty()) {
            makefile << "\t@mv -- " << MakeRecipe(QuoteCceBuildArg(task.publishFrom)) << " "
                     << MakeRecipe(QuoteCceBuildArg(task.outputPath)) << "\n\n";
        }
    }

    makefile << ".PHONY: all clean\n\n";
    makefile << "clean:\n";
    makefile << "\t@rm -f ";
    for (const auto& task : tasks) {
        makefile << MakeRecipe(QuoteCceBuildArg(task.outputPath)) << " ";
        if (!task.publishFrom.empty()) {
            makefile << MakeRecipe(QuoteCceBuildArg(task.publishFrom)) << " ";
        }
    }
    makefile << "\n";

    return makefile.str();
}
} // namespace npu::tile_fwk
