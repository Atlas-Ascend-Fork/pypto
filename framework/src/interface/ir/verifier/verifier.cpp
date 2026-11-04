/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/verifier/verifier.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/error.h"
#include "ir/program.h"

namespace pypto {
namespace ir {

IRVerifier::IRVerifier() = default;

void IRVerifier::AddRule(PropertyVerifierPtr rule)
{
    if (!rule) {
        return;
    }

    // Check if rule with same name already exists
    auto it = std::find_if(rules_.begin(), rules_.end(),
                           [&rule](const PropertyVerifierPtr& r) { return r->GetName() == rule->GetName(); });
    if (it == rules_.end()) {
        rules_.push_back(rule);
    }
}

void IRVerifier::EnableRule(const std::string& name) { disabledRules_.erase(name); }

void IRVerifier::DisableRule(const std::string& name) { disabledRules_.insert(name); }

bool IRVerifier::IsRuleEnabled(const std::string& name) const { return disabledRules_.count(name) == 0; }

std::vector<Diagnostic> IRVerifier::Verify(const ProgramPtr& program) const
{
    if (!program) {
        return {};
    }

    std::vector<Diagnostic> allDiagnostics;

    // Run all enabled verifiers on the program
    // Each verifier internally decides whether to iterate over functions
    for (const auto& rule : rules_) {
        if (!rule) {
            continue;
        }

        // Skip disabled rules
        if (!IsRuleEnabled(rule->GetName())) {
            continue;
        }

        // Run the verifier with the full program
        rule->Verify(program, allDiagnostics);
    }

    return allDiagnostics;
}

void IRVerifier::VerifyOrThrow(const ProgramPtr& program) const
{
    auto diagnostics = Verify(program);

    // Check if there are any errors (not just warnings)
    bool hasErrors = std::any_of(diagnostics.begin(), diagnostics.end(),
                                 [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::ERROR; });
    if (hasErrors) {
        std::string report = GenerateReport(diagnostics);
        throw VerificationError(report, std::move(diagnostics));
    }
}

std::string IRVerifier::GenerateReport(const std::vector<Diagnostic>& diagnostics)
{
    std::ostringstream oss;

    // Count errors and warnings
    size_t errorCount = 0;
    size_t warningCount = 0;
    for (const auto& d : diagnostics) {
        if (d.severity == DiagnosticSeverity::ERROR) {
            errorCount++;
        } else {
            warningCount++;
        }
    }

    // Header
    oss << "IR Verification Report\n";
    oss << "======================\n";
    oss << "Total diagnostics: " << diagnostics.size() << " (";
    oss << errorCount << " errors, " << warningCount << " warnings)\n\n";

    if (diagnostics.empty()) {
        oss << "Status: PASSED\n";
        return oss.str();
    }

    // List all diagnostics
    for (size_t i = 0; i < diagnostics.size(); ++i) {
        const auto& d = diagnostics[i];

        // Severity label
        std::string severityStr = (d.severity == DiagnosticSeverity::ERROR) ? "ERROR" : "WARNING";

        oss << "[" << (i + 1) << "] " << severityStr << " - " << d.ruleName << "\n";
        oss << "  Message: " << d.message << "\n";
        oss << "  Location: " << d.span.Filename() << ":" << d.span.BeginLine() << ":" << d.span.BeginColumn() << "\n";
        oss << "  Error Code: " << d.errorCode << "\n";
        oss << "\n";
    }

    // Summary
    if (errorCount > 0) {
        oss << "Status: FAILED (" << errorCount << " error(s) found)\n";
    } else {
        oss << "Status: PASSED with " << warningCount << " warning(s)\n";
    }

    return oss.str();
}

IRVerifier IRVerifier::CreateDefault()
{
    IRVerifier verifier;

    // Add built-in property verifiers
    verifier.AddRule(CreateSSAPropertyVerifier());
    verifier.AddRule(CreateTypeCheckPropertyVerifier());
    verifier.AddRule(CreateNoNestedCallPropertyVerifier());

    return verifier;
}

} // namespace ir
} // namespace pypto
