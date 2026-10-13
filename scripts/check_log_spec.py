#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the pypto-log-check skill on staged files through a local agent CLI.

The skill is agent-driven, so this hook is a no-op when no agent CLI is
installed (the CI runners).  Only findings on lines added by the current
commit are reported; 致命/严重 findings block the commit.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading

SKILL_FILE = ".agents/skills/pypto-log-check/SKILL.md"
RULES_FILE = ".agents/skills/pypto-log-check/references/log-rules.md"

# 只有这些目录下的源码参与日志规范检查，其余内容一律不扫。
SCAN_DIRS = (
    "framework/include",
    "framework/src",
    "python/pypto",
    "python/pypto_pro",
    "python/src",
)

BEGIN_MARKER = "LOG_CHECK_JSON_BEGIN"
END_MARKER = "LOG_CHECK_JSON_END"

# Probed in this order; cannbot is the CANN-specific agent.  The
# skip-permissions flags let the agent read the reviewed files without an
# interactive prompt; the prompt itself forbids any write.
AGENT_COMMANDS = {
    "cannbot": lambda prompt: ["cannbot", "run", "--dangerously-skip-permissions", prompt],
    "cursor-agent": lambda prompt: ["cursor-agent", "-p", prompt, "--output-format", "text"],
    "claude": lambda prompt: ["claude", "-p", prompt, "--permission-mode", "plan"],
    "codex": lambda prompt: ["codex", "exec", "--sandbox", "read-only", prompt],
    "opencode": lambda prompt: ["opencode", "run", "--dangerously-skip-permissions", prompt],
}
AGENT_ORDER = ["cannbot", "cursor-agent", "claude", "codex", "opencode"]

# Severity levels come from references/log-rules.md; 致命/严重 block the commit.
BLOCKING_SEVERITIES = ("致命", "严重")
SEVERITY_ALIASES = {
    "fatal": "致命",
    "critical": "致命",
    "blocker": "致命",
    "major": "严重",
    "serious": "严重",
    "medium": "中等",
    "moderate": "中等",
    "minor": "提示",
    "info": "提示",
    "hint": "提示",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

DEFAULT_TIMEOUT = 300

# Marker the reviewed agent emits after loading the rules, so the hook can
# relay that phase in real time.
SKILL_LOADED_MARKER = "SKILL_LOADED"
BANNER_RE = re.compile(r"^>\s*\S+\s*·\s*(?P<model>\S+)")

# Broad gate: only pay for an agent call when an added line looks like it could
# be a runtime log/print statement.  Over-inclusive on purpose to avoid misses.
LOG_SIGNAL_RE = re.compile(
    r"(?i)("
    r"log\w*\s*\(|logger\s*\.|logging\s*\.|"
    r"print\w*\s*\(|printf\s*\(|fprintf\s*\(|fputs\s*\(|fwrite\s*\(|"
    r"std::cout|std::cerr|cout\s*<<|cerr\s*<<|console\.log\s*\(|"
    r"warn\w*\s*\(|error\w*\s*\(|fatal\w*\s*\(|trace\w*\s*\(|report\w*\s*\(|dump\w*\s*\("
    r")"
)


def log(message):
    print(f"[log-spec] {message}", file=sys.stderr)


def select_agent():
    """Return the agent name to use, or None (with a reason logged) when
    no agent CLI is available."""
    requested = os.environ.get("PYPTO_LOG_CHECK_AGENT", "").strip()
    if requested:
        if requested not in AGENT_COMMANDS:
            log(f"PYPTO_LOG_CHECK_AGENT={requested} 不是支持的 agent，"
                f"可选：{', '.join(AGENT_ORDER)}；跳过日志规范检查。")
        elif shutil.which(requested) is None:
            log(f"PYPTO_LOG_CHECK_AGENT={requested} 未安装，跳过日志规范检查。")
        else:
            return requested
        return None
    for name in AGENT_ORDER:
        if shutil.which(name) is not None:
            return name
    log(f"未检测到本地 agent CLI（{', '.join(AGENT_ORDER)}），跳过日志规范检查。")
    return None


def in_scan_scope(path):
    """Return True when *path* lives under one of the scanned directories."""
    normalized = path.replace(os.sep, "/")
    if os.path.isabs(normalized):
        normalized = os.path.relpath(normalized).replace(os.sep, "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return any(
        normalized == directory or normalized.startswith(directory + "/")
        for directory in SCAN_DIRS
    )


def staged_added(path):
    """Return (added line numbers, added line texts) for *path*'s staged diff."""
    result = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", path],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = set()
    texts = []
    for line in result.stdout.splitlines():
        match = HUNK_RE.match(line)
        if match:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            lines.update(range(start, start + count))
        elif line.startswith("+") and not line.startswith("+++"):
            texts.append(line[1:])
    return lines, texts


def staged_diff(files, context=4):
    """Return the staged diff (with context) for *files*."""
    result = subprocess.run(
        ["git", "diff", "--cached", f"-U{context}", "--"] + files,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


FINDING_SCHEMA = (
    '{"findings": [{"file": "相对仓库根目录的路径", "line": 行号, '
    '"severity": "致命|严重|中等|提示", "rule": "规则表中的规则名", '
    '"message": "问题描述", "suggestion": "修复建议"}]}'
)


def build_prompt(files, allowed_lines, diff_text):
    index = "\n".join(
        f"- {path}: 新增行 {', '.join(str(n) for n in sorted(allowed_lines[path]))}"
        for path in files
    )
    with open(SKILL_FILE, encoding="utf-8") as handle:
        skill = handle.read()
    with open(RULES_FILE, encoding="utf-8") as handle:
        rules = handle.read()
    return f"""你是 CANN 日志规范评测器，正在 git pre-commit 钩子中运行，工作目录是仓库根目录。

【硬性约束】严禁使用任何工具、严禁读取任何文件、严禁执行任何命令；下面已提供全部所需信息，
请直接完成评测并只输出结果。读取文件会产生额外的多轮调用并拖慢提交，务必避免。

===== 规则（唯一来源 {RULES_FILE}，逐字提供，判定以此为准） =====
{rules}

===== 评测方法（{SKILL_FILE}） =====
{skill}

===== 本次提交：新增行与 staged diff =====
涉及文件及新增行号：
{index}

staged diff（仅新增行需评测）：
```diff
{diff_text}
```

===== 输出 =====
读完规则后先输出一行 {SKILL_LOADED_MARKER}，然后完成评测。
只评测上述新增行中的运行时日志/打印语句，行号使用新文件的绝对行号，遵守规则文件的"全局判定尺度"。
最后只输出一段 JSON，包裹在下面两行标记之间，标记之外不要有其它内容：
{BEGIN_MARKER}
{FINDING_SCHEMA}
{END_MARKER}

没有问题时输出 {{"findings": []}}。
"""


def report_progress(line, state):
    """Relay agent progress markers to the user as each phase completes."""
    text = ANSI_RE.sub("", line).strip()
    if not text:
        return
    if state["model"] is None:
        match = BANNER_RE.match(text)
        if match:
            state["model"] = match.group("model")
            log(f"步骤 2/4：当前模型 = {state['model']}")
            return
    if not state["skill"] and SKILL_LOADED_MARKER in text:
        state["skill"] = True
        log(f"步骤 3/4：skill 加载成功（{SKILL_FILE} + {RULES_FILE}）")
        log(f"步骤 4/4：评测 {state['files']} 个文件 ...")


def run_agent(agent, prompt, timeout, state):
    """Stream the agent output, relaying progress; return it or None on failure."""
    argv = AGENT_COMMANDS[agent](prompt)
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        log(f"{agent} 调用失败（{exc}），跳过日志规范检查。")
        return None

    collected = []

    def reader():
        try:
            for line in process.stdout:
                collected.append(line)
                report_progress(line, state)
        except (OSError, ValueError):
            pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        log(f"{agent} 超过 {timeout}s 未返回，跳过日志规范检查。")
        return None
    finally:
        thread.join(timeout=5)

    if process.returncode != 0:
        detail = "".join(collected).strip().splitlines()
        tail = detail[-1] if detail else "无输出"
        log(f"{agent} 返回码 {process.returncode}（{tail}），跳过日志规范检查。")
        return None
    return "".join(collected)


def parse_findings(output):
    """Return the findings list from the agent output, or None if unparsable."""
    text = ANSI_RE.sub("", output or "")
    start = text.rfind(BEGIN_MARKER)
    end = text.rfind(END_MARKER)
    if start != -1 and end > start:
        payload = text[start + len(BEGIN_MARKER):end]
    else:
        brace = text.find('{"findings"')
        if brace == -1:
            brace = text.find('{ "findings"')
        if brace == -1:
            return None
        payload = text[brace:text.rfind("}") + 1]
    try:
        data = json.loads(payload.strip())
    except json.JSONDecodeError:
        return None
    findings = data.get("findings") if isinstance(data, dict) else None
    return findings if isinstance(findings, list) else None


def normalize(finding, allowed_lines):
    """Return a printable finding, or None when it is out of scope."""
    if not isinstance(finding, dict):
        return None
    path = str(finding.get("file", "")).strip()
    if path not in allowed_lines:
        return None
    try:
        line = int(finding.get("line"))
    except (TypeError, ValueError):
        return None
    if line not in allowed_lines[path]:
        return None
    severity = str(finding.get("severity", "")).strip()
    severity = SEVERITY_ALIASES.get(severity.lower(), severity)
    return {
        "file": path,
        "line": line,
        "severity": severity,
        "rule": str(finding.get("rule", "")).strip(),
        "message": str(finding.get("message", "")).strip(),
        "suggestion": str(finding.get("suggestion", "")).strip(),
    }


def report(findings):
    """Print the findings and return the number of blocking ones."""
    blocking = 0
    for item in sorted(findings, key=lambda f: (f["file"], f["line"])):
        log(f"{item['severity']} {item['file']}:{item['line']} "
            f"[{item['rule']}] {item['message']}")
        if item["suggestion"]:
            log(f"         建议：{item['suggestion']}")
        if item["severity"] in BLOCKING_SEVERITIES:
            blocking += 1
    return blocking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    if os.environ.get("PYPTO_SKIP_LOG_CHECK"):
        log("PYPTO_SKIP_LOG_CHECK 已设置，跳过日志规范检查。")
        return 0
    if not args.files:
        return 0
    files = [path for path in args.files if in_scan_scope(path)]
    if not files:
        return 0
    if not os.path.exists(RULES_FILE) or not os.path.exists(SKILL_FILE):
        log("未找到规则或 skill 文件，跳过日志规范检查。")
        return 0

    log("步骤 1/4：检测本地 agent CLI ...")
    agent = select_agent()
    if agent is None:
        return 0

    staged = {path: staged_added(path) for path in files}
    staged = {path: value for path, value in staged.items()
              if value[0] and any(LOG_SIGNAL_RE.search(text) for text in value[1])}
    if not staged:
        log("本次新增行未检测到日志/打印语句，跳过 agent 评测。")
        return 0

    files = sorted(staged)
    allowed_lines = {path: lines for path, (lines, _) in staged.items()}
    timeout = int(os.environ.get("PYPTO_LOG_CHECK_TIMEOUT", DEFAULT_TIMEOUT))
    log(f"步骤 1/4：agent = {agent} 可用，待评测 {len(files)} 个文件，最长等待 {timeout}s")

    diff_text = staged_diff(files)
    state = {"model": None, "skill": False, "files": len(files)}
    output = run_agent(agent, build_prompt(files, allowed_lines, diff_text), timeout, state)
    if output is None:
        return 0
    if not state["skill"]:
        log("提示：未捕获到 skill 加载标记，agent 可能未按步骤加载 skill。")

    findings = parse_findings(output)
    if findings is None:
        log(f"{agent} 输出无法解析为评测结果，跳过日志规范检查。")
        return 0

    in_scope = [f for f in (normalize(f, allowed_lines) for f in findings) if f]
    if not in_scope:
        log("本次改动的日志语句未发现规范问题。")
        return 0

    blocking = report(in_scope)
    if blocking == 0:
        log(f"发现 {len(in_scope)} 个非阻断级问题（中等/提示），不影响提交。")
        return 0
    log(f"发现 {blocking} 个阻断级问题（{'/'.join(BLOCKING_SEVERITIES)}），提交已拦截。")
    log("确认为误报时可用 SKIP=log-spec-check git commit ... 跳过本检查。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
