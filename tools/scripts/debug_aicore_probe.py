#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""
步骤 3.5: CCE / machine AICore 框架 行级打点与落盘判读（debug_aicore_probe）

用 debug_aicore_error.py 的输出（<out>/info_<ts>/info.txt）作为输入：
  1. 从 section 3 "Operator Error Line Number" 解析全部 Corrected Info；
     优先选 AIC_ERROR != 0 且 outerSrc 非入口桩（aicore_<hash>.cpp）的记录；
     若 outerSrc 是入口桩且 innerSrc 落在 tilefwk/aicore_*.h，则走【框架模式】，
     对已安装（或源码树）头文件插桩，而不是改入口桩 CCE。
  2. 在目标行前后 window 行内的每个"安全语句边界"后插入一行
     PYPTO_AICORE_PRINTF("[PROBE:<tag>] <文件名> L=<原行号> | <该行源码摘要>");
     其中 L= 始终是插桩前（.bak）行号。
  3. 提示用户重新执行用例；框架模式还需删入口桩 .o 以触发带新头的重编。

模式（三者互斥，必须显式指定其一）:
  --probe    插桩
  --analyze  解析 printf 落盘 bin，推断最后到达行
  --restore  从 .bak 还原

用法:
    # 插桩：自动分流（CCE / 框架）
    python3 debug_aicore_probe.py --probe --report-dir <debug_aicore_error.py 的 -out 目录>

    # 插桩：强制框架头（装包路径）
    python3 debug_aicore_probe.py --probe --header /path/to/aicore_entry_devtask.h --line 115

    # 插桩：手工 CCE
    python3 debug_aicore_probe.py --probe --cce <file.cpp> --line 30

    # 落盘判读（--log-dir 须与复现时的 ASCEND_WORK_PATH 一致；省略则读 $ASCEND_WORK_PATH。
    #   多核取各核 last L 最小值）
    python3 debug_aicore_probe.py --analyze --log-dir "$ASCEND_WORK_PATH" --tag T1

    # 还原（只按路径找 .bak，不需要 --line；框架模式优先用插桩时的绝对路径）
    python3 debug_aicore_probe.py --restore --header /path/to/aicore_entry_devtask.h
    python3 debug_aicore_probe.py --restore --report-dir <...>

返回值:
    Exit 0: 成功 / Exit 1: 失败 / Exit 130: Ctrl+C
"""

import argparse
import glob
import logging
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

setup_logging = None  # 允许独立运行时不依赖 common.py
try:
    from common import setup_logging  # noqa: F401
except Exception:  # pragma: no cover - 独立运行兜底
    pass

if setup_logging is None:
    def setup_logging():
        logging.basicConfig(level=logging.INFO, format='%(message)s')

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_SECTION3_TITLE = "Operator Error Line Number"
_SECTION_HEADER_RE = re.compile(r"^\*{3,}\s*\d+\.\s+.+?\*{3,}\s*$", re.MULTILINE)
# outerSrc: /path/to/xxx.cpp:30:12  （path 可能含 ':'，用贪婪匹配吃掉，末尾两个数字为行、列）
_OUTER_SRC_RE = re.compile(r"^outerSrc:\s*(.+?):(\d+):(\d+)\s*$", re.MULTILINE)
_INNER_SRC_RE = re.compile(r"^innerSrc:\s*(.+?):(\d+):(\d+)\s*$", re.MULTILINE)
_KERNEL_NAME_RE = re.compile(r"^kernel name\s+:\s*(.+?)\s*$", re.MULTILINE)
# section 3 里每条 Corrected Info 前都有 AIC_ERROR / core id
_AIC_ERROR_RE = re.compile(r"^AIC_ERROR\s*:\s*(0x[0-9a-fA-F]+|\d+)\s*$", re.MULTILINE)
_CORE_ID_RE = re.compile(r"^core id\((aic|aiv)\)\s*:\s*(.+?)\s*$", re.MULTILINE)

# 入口桩文件名形如 aicore_<hash>.cpp，里面只有 KERNEL_ENTRY → KernelEntry 转发，插桩无意义
_ENTRY_STUB_RE = re.compile(r"^aicore_\d+\.cpp$")

# machine AICore 框架头（innerSrc 常见落点）
_FRAMEWORK_HEADER_RE = re.compile(
    r"(?:^|/)(aicore_entry(?:_drco|_devtask|_dfx)?\.h|aicore_runtime\.h)$"
)

# 向量/汇编作用域内不能插桩：printf 是标量操作，插进去会破坏向量语义。
# 注意：不要把 __aicore__ 算进去——它只是函数/修饰符，框架头几乎每行都有，
# 误判会导致 safe_points 在 aicore_entry_*.h 里几乎为空。
_UNSAFE_SCOPE_MARKERS = ("__VEC_SCOPE__", "__ASM_CODE__", "__ASM__")

_PROBE_INCLUDE = '#include "tilefwk/aicore_pto_debug.h"'

# aicore_pto_debug.h 里 get_sys_cnt() 若未加命名空间，开 print 会编译失败
_BAD_SYS_CNT = "static_cast<uint32_t>(core_id), get_sys_cnt(), ##__VA_ARGS__)"
_GOOD_SYS_CNT = "static_cast<uint32_t>(core_id), __cce_scalar::get_sys_cnt(), ##__VA_ARGS__)"

# [PROBE:T1] foo.cpp|.h L=74 | ...
# bin 里可能仍带 CANN 前缀（未做 host 侧格式化）
_PROBE_HIT_RE = re.compile(
    rb"\[PROBE:([^\]]+)\]\s+(\S+\.(?:cpp|h|hpp))\s+L=(\d+)\s*\|\s*([^\x00\r\n]*)"
)


# ---------------------------------------------------------------------------
# 前置提示
# ---------------------------------------------------------------------------

def _precheck_pause(wait_seconds: int, mode: str = "cce"):
    """按插桩模式打印前置条件。framework 与 cce 要求不同，勿混用 CCE 口径。"""
    logger.info("=" * 78)
    logger.info(f"前置条件检查（模式={mode}；不满足则插桩无效）")
    logger.info("=" * 78)
    if mode == "framework":
        logger.info("  【框架头】挂点在 tilefwk/aicore_*.h（装包或源码头），不是业务算子 CCE。")
        logger.info("  1) 已用 debug_aicore_error.py 产出 info.txt，且 innerSrc 指向框架头")
        logger.info("  2) 插桩改的是头文件；复现前删除入口桩目标文件以触发带新头重编：")
        logger.info("     —— rm -f <kernel_aicore>/aicore_*.o   （不要删 .cpp / 子函数 CCE）")
        logger.info("  3) 测完务必还原头文件：")
        logger.info("     —— python3 debug_aicore_probe.py --restore --header <本次插桩的绝对路径>")
        logger.info("")
        logger.info("  说明：compile_debug_mode=2 /「勿删 kernel_aicore」是 CCE 模式要求，框架头模式不适用。")
        abort_hint = "已中止：请确认 info.txt / 装包头路径后，再重新执行 --probe。"
    else:
        logger.info("  【CCE】挂点在生成的算子/子函数 CCE（*.cpp）。")
        logger.info("  1) 已固定 CCE 输出路径并复现问题")
        logger.info("     —— 用例需使用 compile_debug_mode=2（fixed CCE，不覆盖已有 CCE 文件）")
        logger.info("  2) 已用 debug_aicore_error.py 产出 info.txt")
        logger.info(
            "     —— python {pypto}/lib/scripts/debug_aicore_error.py "
            "-p <ASCEND_WORK_PATH或用例plog目录> -out ./diagnosis -d <id>"
        )
        logger.info("  3) 重跑用例时【不要删除 kernel_aicore 目录】，否则本次插桩会被覆盖")
        abort_hint = "已中止：请先固定 CCE 并复现问题，再重新执行本脚本。"
    logger.info("")
    logger.info("  如果以上条件尚未满足，请按 Ctrl+C 中止。")
    logger.info("=" * 78)

    if wait_seconds <= 0:
        return
    try:
        for i in range(wait_seconds, 0, -1):
            sys.stdout.write(f"\r  {i:2d} 秒后继续插桩，如需中止请按 Ctrl+C ...")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        logger.info(abort_hint)
        sys.exit(130)


# ---------------------------------------------------------------------------
# info.txt 解析
# ---------------------------------------------------------------------------

def _find_info_txt(report_dir: str) -> str:
    """在 debug_aicore_error.py 的 -out 目录下递归找最新的 info.txt。"""
    candidates = glob.glob(os.path.join(report_dir, "**", "info.txt"), recursive=True)
    if not candidates:
        # 允许直接传 info.txt 本身或 info_<ts> 目录
        for name in ("info.txt", os.path.join("**", "info.txt")):
            candidates += glob.glob(os.path.join(report_dir, name), recursive=True)
    if not candidates:
        raise RuntimeError(
            f"在 {report_dir} 下未找到 info.txt；请确认传入的是 debug_aicore_error.py 的 -out 目录"
        )
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if len(candidates) > 1:
        logger.info(f"发现 {len(candidates)} 个 info.txt，取最新的一个：{candidates[0]}")
    return candidates[0]


def _extract_section(content: str, title_keyword: str) -> str:
    """按 section header 切分，返回标题含 title_keyword 的段内容。"""
    matches = list(_SECTION_HEADER_RE.finditer(content))
    for i, m in enumerate(matches):
        if title_keyword in m.group():
            start = m.end() + 1
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            return content[start:end]
    return ""


def _parse_aic_error_code(raw: str) -> int:
    raw = raw.strip().lower()
    return int(raw, 16) if raw.startswith("0x") else int(raw)


def _is_entry_stub_path(path: str) -> bool:
    return bool(_ENTRY_STUB_RE.match(os.path.basename(path.strip())))


def _is_framework_header(path: str) -> bool:
    if not path:
        return False
    base = os.path.basename(path.strip())
    if _FRAMEWORK_HEADER_RE.search(base):
        return True
    # 宽松：tilefwk 下 aicore_*.h（排除 print/dump 纯声明头也可插，由调用方决定）
    norm = path.replace("\\", "/")
    return "/tilefwk/" in norm and base.startswith("aicore_") and base.endswith(".h")


def _parse_section3_records(sec3: str):
    """把 section 3 拆成多条 {aic_error, core_kind, outer, inner} 记录。"""
    starts = [m.start() for m in _AIC_ERROR_RE.finditer(sec3)]
    if not starts:
        # 兼容旧格式：没有 AIC_ERROR 头时，退回按 outerSrc 切
        starts = [m.start() for m in _OUTER_SRC_RE.finditer(sec3)]
    if not starts:
        return []

    records = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(sec3)
        block = sec3[start:end]
        err_m = _AIC_ERROR_RE.search(block)
        outer_m = _OUTER_SRC_RE.search(block)
        if not outer_m:
            continue
        inner_m = _INNER_SRC_RE.search(block)
        core_m = _CORE_ID_RE.search(block)
        cce_path = outer_m.group(1).strip()
        inner_path = inner_m.group(1).strip() if inner_m else None
        inner_line = int(inner_m.group(2)) if inner_m else None
        inner_col = int(inner_m.group(3)) if inner_m else None
        records.append({
            "aic_error": _parse_aic_error_code(err_m.group(1)) if err_m else None,
            "core_kind": core_m.group(1) if core_m else None,
            "core_id": core_m.group(2).strip() if core_m else None,
            "cce_path": cce_path,
            "line": int(outer_m.group(2)),
            "col": int(outer_m.group(3)),
            "inner_path": inner_path,
            "inner_line": inner_line,
            "inner_col": inner_col,
            "inner_src": (
                f"{inner_path}:{inner_line}:{inner_col}" if inner_path else None
            ),
            "is_stub": _is_entry_stub_path(cce_path),
            "inner_is_framework": _is_framework_header(inner_path or ""),
        })
    return records


def _select_fault_record(records):
    """
    从多条 Corrected Info 里挑真正该插桩的那条。

    优先级：
      0) AIC_ERROR != 0 且 outerSrc 不是入口桩（业务 CCE 真故障）
      1) outerSrc 不是入口桩
      2) AIC_ERROR != 0 且 innerSrc 是框架头（入口桩 + 框架挂点，走 framework 模式）
      3) innerSrc 是框架头
      4) 其余入口桩 / 调度超时 —— 兜底
    """
    if not records:
        return None, []

    def rank(rec):
        err = rec["aic_error"]
        real_err = err is not None and err != 0
        fw = rec.get("inner_is_framework")
        if real_err and not rec["is_stub"]:
            return 0
        if not rec["is_stub"]:
            return 1
        if real_err and fw:
            return 2
        if fw:
            return 3
        return 4

    best = min(enumerate(records), key=lambda ir: (rank(ir[1]), ir[0]))[1]
    skipped = [r for r in records if r is not best]
    return best, skipped


def _fmt_record(rec) -> str:
    err = rec["aic_error"]
    err_s = f"0x{err:x}" if isinstance(err, int) else "?"
    core = rec["core_kind"] or "?"
    stub = " [入口桩]" if rec["is_stub"] else ""
    fw = " [框架inner]" if rec.get("inner_is_framework") else ""
    inner = f" inner={rec['inner_src']}" if rec.get("inner_src") else ""
    return (
        f"AIC_ERROR={err_s} core={core} "
        f"{rec['cce_path']}:{rec['line']}:{rec['col']}{stub}{fw}{inner}"
    )


def parse_info_txt(info_path: str):
    """解析 info.txt，返回 (cce_path, line, col, extra_info)。

    mix 核常见同一 info 里既有 AIC 调度超时（落在 aicore_<hash>.cpp / KernelEntry），
    又有 AIV/AIC 业务错误（落在 *_aiv_main.cpp / *_aic_main.cpp）。本函数扫完全部
    Corrected Info，优先返回后者。
    """
    with open(info_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    sec3 = _extract_section(content, _SECTION3_TITLE)
    if not sec3:
        raise RuntimeError(f'section 3 "{_SECTION3_TITLE}" not found in {info_path}')

    records = _parse_section3_records(sec3)
    if not records:
        raise RuntimeError("outerSrc not found in section 3, 无法确定可疑行")

    best, skipped = _select_fault_record(records)
    if len(records) > 1:
        logger.info(f"section 3 共 {len(records)} 条 Corrected Info，按优先级选择：")
        logger.info(f"  [选用] {_fmt_record(best)}")
        for rec in skipped:
            logger.info(f"  [跳过] {_fmt_record(rec)}")

    extra = {
        "innerSrc": best["inner_src"],
        "inner_path": best.get("inner_path"),
        "inner_line": best.get("inner_line"),
        "inner_is_framework": best.get("inner_is_framework"),
        "aic_error": best["aic_error"],
        "core_kind": best["core_kind"],
        "is_stub": best["is_stub"],
        "all_records": records,
    }
    mk = _KERNEL_NAME_RE.search(content)
    if mk:
        extra["kernel"] = mk.group(1)

    return best["cce_path"], best["line"], best["col"], extra


def _suggest_subfunc_cce(cce_path: str):
    """outerSrc 指向入口桩时，列出同目录下真正的子函数 CCE 作为候选。"""
    cce_dir = os.path.dirname(cce_path)
    if not os.path.isdir(cce_dir):
        return []
    cands = [
        p for p in glob.glob(os.path.join(cce_dir, "*.cpp"))
        if not _ENTRY_STUB_RE.match(os.path.basename(p))
    ]
    return sorted(cands)


# ---------------------------------------------------------------------------
# 环境自检：两个已知的"开了 print 反而编译不过"的坑
# ---------------------------------------------------------------------------

def _installed_include_dir():
    """返回已安装 pypto 的 tilefwk 头目录。"""
    import subprocess

    loc = None
    for cmd in (["pip3", "show", "pypto"], [sys.executable, "-m", "pip", "show", "pypto"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        if r.returncode != 0:
            continue
        for ln in r.stdout.splitlines():
            if ln.startswith("Location:"):
                loc = ln.split(":", 1)[1].strip()
                break
        if loc:
            break

    if not loc:
        # 兜底：直接从当前解释器的 site-packages 推断
        for d in sys.path:
            cand = os.path.join(d, "pypto", "lib", "include", "tilefwk")
            if os.path.isdir(cand):
                loc = d
                break
    if not loc:
        return None
    p = os.path.join(loc, "pypto", "lib", "include", "tilefwk")
    return p if os.path.isdir(p) else None


def _patch_once(path: str, old: str, new: str, desc: str) -> bool:
    """幂等打补丁：命中 old 才替换，首次修改前留 .bak。"""
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    if old not in src:
        return False
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new))
    logger.info(f"  [已修复] {desc}")
    logger.info(f"           文件: {path}（原文件备份为 .bak）")
    return True


def env_selfcheck(auto_patch: bool = True) -> bool:
    """
    检查 PYPTO_AICORE_PRINTF（pto_debug）在本机是否可用。返回是否已就绪。

    仅认 tilefwk/aicore_pto_debug.h + tilefwk/pto_debug/；不再兼容旧 aicore_asc_*。
    若 aicore_pto_debug.h 中 get_sys_cnt() 未加 __cce_scalar::，可自动热补丁。
    """
    logger.info("-" * 78)
    logger.info("环境自检：pto_debug print 通道")

    inc = _installed_include_dir()
    if inc is None:
        logger.info("  [跳过] 未找到已安装 pypto 的 tilefwk 头目录")
        return True

    pto_debug_h = os.path.join(inc, "aicore_pto_debug.h")
    pto_debug_root = os.path.join(inc, "pto_debug")
    if not os.path.isfile(pto_debug_h):
        logger.info("  [警告] 未找到 aicore_pto_debug.h，当前 pypto 版本可能不支持 PYPTO_AICORE_PRINTF")
        return False

    ok = True
    printf_impl = os.path.join(pto_debug_root, "impl", "utils", "debug", "pto_aicore_printf_impl.h")
    dump_h = os.path.join(pto_debug_root, "include", "utils", "debug", "pto_dump.h")
    if not (os.path.isfile(printf_impl) and os.path.isfile(dump_h)):
        logger.info("  [警告] tilefwk/pto_debug/ 不完整（缺 pto_aicore_printf_impl.h 或 pto_dump.h）")
        ok = False
    else:
        logger.info("  [正常] 找到 aicore_pto_debug.h 与 pto_debug 实现树")

    body = open(pto_debug_h, encoding="utf-8", errors="replace").read()
    if _BAD_SYS_CNT in body:
        if auto_patch:
            _patch_once(pto_debug_h, _BAD_SYS_CNT, _GOOD_SYS_CNT,
                        "aicore_pto_debug.h: get_sys_cnt() -> __cce_scalar::get_sys_cnt()")
        else:
            logger.info("  [未修复] aicore_pto_debug.h 中 get_sys_cnt() 缺命名空间限定，编译会失败")
            ok = False

    if ok:
        logger.info("  [就绪] PYPTO_AICORE_PRINTF（pto）可用")
    return ok


# ---------------------------------------------------------------------------
# 插桩
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    idx = line.find("//")
    return line[:idx] if idx != -1 else line


def _paren_balanced(s: str) -> bool:
    s = re.sub(r'"(\\.|[^"\\])*"', '""', s)
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _scope_kind_for_open(look: str) -> str:
    """粗分 `{` 打开的作用域：fn / ns / class / enum / ctrl / other。"""
    look = look.strip()
    if re.search(r"\bnamespace\b", look):
        return "ns"
    if re.search(r"\benum\b", look):
        return "enum"
    if re.search(r"\b(class|struct|union)\b", look):
        # 返回类类型的函数：`) ... {`；成员函数定义同理
        if re.search(r"\)\s*(const|override|final|noexcept|volatile|&)?\s*$", look) or look.endswith(")"):
            return "fn"
        return "class"
    if re.match(r"^(else|try|catch|do)\b", look) or re.match(r"^(if|for|while|switch)\b", look):
        return "ctrl"
    if look.endswith(")") or re.search(r"\)\s*(const|override|final|noexcept|->)\b", look):
        return "fn"
    return "other"


def _lookback_for_brace(lines, i, max_back: int = 5) -> str:
    for j in range(i - 1, max(0, i - max_back) - 1, -1):
        c = _strip_comment(lines[j]).strip()
        if c and not c.startswith("#"):
            return c
    return ""


def _apply_braces(stack: list, raw_s: str, open_kind: str) -> None:
    """按字符更新作用域栈；同一行多个 `{` 时仅第一个用 open_kind。"""
    kind = open_kind
    for ch in raw_s:
        if ch == "{":
            stack.append(kind)
            kind = "other"
        elif ch == "}":
            if stack:
                stack.pop()


def safe_points(lines):
    """
    返回可以在'其后'插入打印的行号（1-based）。

    规则：必须在函数体内且插桩后仍在函数体内（否则打印落到 namespace/class/文件作用域会编译报错）、
    非预处理指令、语句完整（以 ; 结尾且括号配平，或单独 } 结束一个块）、
    不在 __VEC_SCOPE__ / __ASM__ 内、不在 return/break/continue 之后。
    """
    pts = []
    stack = []  # 元素: fn / ns / class / enum / ctrl / other
    unsafe_until_depth = None

    for i, raw in enumerate(lines):
        code = _strip_comment(raw).strip()
        raw_s = raw.strip()
        depth = len(stack)

        open_kind = "other"
        if "{" in raw_s:
            prefix = code.split("{", 1)[0].strip() if code else ""
            look = prefix or _lookback_for_brace(lines, i)
            open_kind = _scope_kind_for_open(look)

        if not code or code.startswith("#"):
            _apply_braces(stack, raw_s, open_kind)
            continue

        if unsafe_until_depth is not None:
            if code == "}" and depth <= unsafe_until_depth:
                unsafe_until_depth = None
            _apply_braces(stack, raw_s, open_kind)
            continue
        if any(m in code for m in _UNSAFE_SCOPE_MARKERS):
            unsafe_until_depth = depth
            _apply_braces(stack, raw_s, open_kind)
            continue

        depth_before = depth
        in_function = "fn" in stack
        _apply_braces(stack, raw_s, open_kind)
        depth_after = len(stack)

        if depth_before < 1 or depth_after < 1 or not in_function:
            continue
        is_stmt = code.endswith(";") and _paren_balanced(code)
        is_block_end = code == "}"
        if not (is_stmt or is_block_end):
            continue
        if re.match(r"^(return|break|continue|goto)\b", code):
            continue

        pts.append(i + 1)

    return pts


def _indent_of(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _probe_src_snippet(raw: str, max_len: int = 80) -> str:
    """把原文件一行压成可放进 printf 字符串字面量的短摘要。

    L= 对应插桩前行号；摘要让用户看日志即可对照源码，无需打开 .bak。
    作为 printf 的 fmt 字面量：转义 \\ \" ，并把 % 写成 %%，避免被当格式符。
    """
    s = raw.strip()
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    # 控制字符压成空格，避免破坏日志行
    s = "".join(ch if 32 <= ord(ch) < 127 or ord(ch) > 127 else " " for ch in s)
    return s


def _decide_mode(args, outer_path: str, extra: dict) -> str:
    """返回 'cce' 或 'framework'。"""
    mode = getattr(args, "mode", "auto") or "auto"
    if mode in ("cce", "framework"):
        return mode
    # auto
    if getattr(args, "header", None):
        return "framework"
    if getattr(args, "cce", None) and not getattr(args, "header", None):
        return "cce"
    if extra.get("is_stub") and extra.get("inner_is_framework"):
        return "framework"
    if _is_entry_stub_path(outer_path) and _is_framework_header(
        (extra.get("inner_path") or "")
    ):
        return "framework"
    return "cce"


def _resolve_probe_target(args):
    """
    解析插桩目标，返回 dict:
      mode, path, line, outer_path, extra
    """
    extra = {}
    outer_path, outer_line, outer_col = None, None, None

    if args.header:
        if args.line is None:
            raise RuntimeError("使用 --header 时必须同时指定 --line")
        return {
            "mode": "framework",
            "path": os.path.abspath(args.header),
            "line": args.line,
            "outer_path": os.path.abspath(args.cce) if args.cce else None,
            "extra": {},
        }

    if args.cce and args.mode == "cce":
        if args.line is None:
            raise RuntimeError("使用 --cce 时必须同时指定 --line")
        return {
            "mode": "cce",
            "path": os.path.abspath(args.cce),
            "line": args.line,
            "outer_path": os.path.abspath(args.cce),
            "extra": {},
        }

    if args.cce and not args.report_dir and not args.info:
        # 手工 CCE，未强制 framework
        if args.line is None:
            raise RuntimeError("使用 --cce 时必须同时指定 --line")
        mode = "framework" if args.mode == "framework" else "cce"
        if mode == "framework":
            raise RuntimeError("框架模式请用 --header <aicore_*.h> --line，或提供 info.txt")
        return {
            "mode": "cce",
            "path": os.path.abspath(args.cce),
            "line": args.line,
            "outer_path": os.path.abspath(args.cce),
            "extra": {},
        }

    info_path = args.info or _find_info_txt(args.report_dir)
    logger.info(f"输入 info.txt: {info_path}")
    outer_path, outer_line, outer_col, extra = parse_info_txt(info_path)
    if outer_col:
        logger.info(f"outerSrc: {outer_path}:{outer_line}:{outer_col}")
    if extra.get("kernel"):
        logger.info(f"kernel name: {extra['kernel']}")
    if extra.get("aic_error") is not None:
        logger.info(f"AIC_ERROR: 0x{extra['aic_error']:x} (core={extra.get('core_kind') or '?'})")
    if extra.get("innerSrc"):
        logger.info(f"innerSrc: {extra['innerSrc']}")

    # 把 parse 结果里的 inner 字段补到 extra（parse_info_txt 已写 innerSrc 字符串）
    # 从 all_records / 再解析一次路径
    best = None
    for rec in extra.get("all_records") or []:
        if rec.get("cce_path") == outer_path and rec.get("line") == outer_line:
            best = rec
            break
    if best:
        extra["inner_path"] = best.get("inner_path")
        extra["inner_line"] = best.get("inner_line")
        extra["inner_is_framework"] = best.get("inner_is_framework")

    mode = _decide_mode(args, outer_path, extra)
    if mode == "framework":
        hpath = extra.get("inner_path")
        hline = extra.get("inner_line")
        if not hpath or hline is None:
            raise RuntimeError(
                "已判定为框架模式，但 info.txt 无可用 innerSrc；"
                "请改用 --header <path> --line <n>"
            )
        if not _is_framework_header(hpath) and args.mode != "framework":
            raise RuntimeError(f"innerSrc 不像框架头: {hpath}")
        return {
            "mode": "framework",
            "path": os.path.abspath(hpath),
            "line": int(hline),
            "outer_path": os.path.abspath(outer_path) if outer_path else None,
            "extra": extra,
        }

    # CCE 模式
    return {
        "mode": "cce",
        "path": os.path.abspath(outer_path),
        "line": int(outer_line),
        "outer_path": os.path.abspath(outer_path),
        "extra": extra,
    }


def instrument_file(path: str, line: int, window: int, tag: str, core0_only: bool = False) -> list:
    """
    在 path 的 line±window 安全语句边界后插入 PROBE。
    返回打点原行号列表。失败抛 RuntimeError。
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise RuntimeError(f"目标文件不存在: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    bak = path + ".bak"
    if "[PROBE:" in "".join(lines):
        if not os.path.exists(bak):
            raise RuntimeError(
                f"文件已有 [PROBE 打点，但找不到 .bak: {bak}；请手工清理或从版本库恢复"
            )
        shutil.copy2(bak, path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        logger.info("检测到已有打点，已先从 .bak 还原为原始文件，再重新插桩")
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        logger.info(f"已备份原文件 -> {bak}")

    pts = safe_points(lines)
    lo = max(1, line - window)
    hi = min(len(lines), line + window)
    selected = [p for p in pts if lo <= p <= hi]
    if not selected:
        raise RuntimeError(
            f"窗口 [{lo}, {hi}] 内没有可插桩的语句边界；"
            f"文件内可插桩点共 {len(pts)} 处：{pts[:40]}"
        )

    body = "".join(lines)
    if _PROBE_INCLUDE not in body:
        # 框架头通常经 aicore_entry.h 间接可见宏；生成 CCE 应自带 include
        logger.info(
            "提示：文件未直接包含 aicore_pto_debug.h；"
            "若为框架头且入口已 include，一般仍可编译"
        )

    fname = os.path.basename(path)
    built = []
    for i, raw in enumerate(lines):
        built.append(raw)
        if (i + 1) in selected:
            orig_ln = i + 1
            snippet = _probe_src_snippet(raw)
            indent = _indent_of(raw)
            built.append(f"{indent}// [PROBE:{tag}] after orig L={orig_ln}: {raw.strip()[:80]}\n")
            msg = f"[PROBE:{tag}] {fname} L={orig_ln} | {snippet}"
            stmt = f'PYPTO_AICORE_PRINTF("{msg}");'
            if core0_only:
                stmt = f"if (PYPTO_AICORE_PRINTF_CORE_ID() == 0) {{ {stmt} }}"
            built.append(indent + stmt + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(built)
    return sorted(selected)


def do_probe(args, resolved_target=None) -> int:
    try:
        target = resolved_target if resolved_target is not None else _resolve_probe_target(args)
    except RuntimeError as e:
        logger.info(f"错误：{e}")
        return 1

    mode = target["mode"]
    path = target["path"]
    line = target["line"]
    outer_path = target.get("outer_path")
    extra = target.get("extra") or {}

    logger.info("")
    logger.info(f"插桩模式: {mode}")
    logger.info(f"目标文件: {path}:{line}")
    if extra.get("aic_error") == 0:
        logger.info("提示: AIC_ERROR=0x0（多为 timeout/trap）。PC 常落在框架超时检查点，")
        logger.info("      插桩可确认卡在哪段 wait；根因可能是调度未推进而非该行本身。")

    if mode == "cce":
        if not os.path.isfile(path):
            logger.info(f"错误：CCE 文件不存在: {path}")
            logger.info("      可能是 CCE 输出目录已被清理；请重新复现问题生成 CCE 后再执行本脚本。")
            return 1
        if _is_entry_stub_path(path):
            logger.info("")
            logger.info("[警告] 目标仍是入口桩 aicore_<hash>.cpp，插桩无法定位真实挂点。")
            if extra.get("inner_is_framework") or _is_framework_header(extra.get("inner_path") or ""):
                logger.info("       检测到 innerSrc 为框架头，请改用 --mode framework，或依赖 --mode auto。")
            cands = _suggest_subfunc_cce(path)
            if cands:
                logger.info("       同目录子函数 CCE 候选：")
                for p in cands:
                    logger.info(f"         {p}")
            if not args.force:
                logger.info("已中止：改用 --cce <子函数> / --mode framework，或加 --force。")
                return 1

    if mode == "framework" and not os.path.isfile(path):
        logger.info(f"错误：框架头不存在: {path}")
        logger.info("      请确认 info.txt 的 innerSrc 路径在本机仍有效（重装 pypto 后路径可能变）。")
        return 1

    try:
        selected = instrument_file(
            path, line, args.window, args.tag, core0_only=args.core0_only
        )
    except RuntimeError as e:
        logger.info(f"错误：{e}")
        return 1

    logger.info("")
    logger.info(f"插桩完成: {path}")
    logger.info(f"  模式 {mode}，目标行 {line}，窗口 ±{args.window}，标签 [{args.tag}]")
    logger.info(f"  打点原行号（{len(selected)} 处）: {selected}")
    logger.info("  日志格式: [PROBE:...] <file> L=<原行号> | <该行源码摘要>")
    logger.info("")
    _print_next_steps(args, path, args.tag, mode=mode, outer_path=outer_path)
    return 0


def _default_work_path(args):
    """解析工作目录：--log-dir > $ASCEND_WORK_PATH；二者皆无则返回 None。

    printf 落在 $ASCEND_WORK_PATH/printf/；--analyze / 下一步提示都依赖此路径，
    须与复现用例时 export 的 ASCEND_WORK_PATH 一致。
    """
    return getattr(args, "log_dir", None) or os.environ.get("ASCEND_WORK_PATH") or None


def _find_latest_printf_dir(work_path: str):
    """定位 $ASCEND_WORK_PATH/printf/<timestamp>/（CANN asc_printf 落盘目录）。"""
    printf_root = os.path.join(os.path.abspath(work_path), "printf")
    if not os.path.isdir(printf_root):
        return None
    subdirs = [
        os.path.join(printf_root, name)
        for name in os.listdir(printf_root)
        if os.path.isdir(os.path.join(printf_root, name))
    ]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def _iter_probe_hits_from_bin(bin_path: str, tag=None):
    """从 asc_kernel_data_*.bin 里扫出 PROBE 命中（按文件内出现顺序）。"""
    try:
        with open(bin_path, "rb") as f:
            data = f.read()
    except OSError:
        return
    for m in _PROBE_HIT_RE.finditer(data):
        hit_tag = m.group(1).decode("utf-8", "replace")
        if tag and hit_tag != tag:
            continue
        yield {
            "tag": hit_tag,
            "file": m.group(2).decode("utf-8", "replace"),
            "line": int(m.group(3)),
            "snippet": m.group(4).decode("utf-8", "replace").rstrip(),
            "bin": os.path.basename(bin_path),
        }


def do_analyze(args) -> int:
    """从 printf 落盘 bin 自动闭环：按 core/bin 取最后一条 PROBE，推断挂点窗口。

    启发式（多核 / 热循环）：
      异常核往往提前停在故障行附近，健康核仍能跑到窗口末尾。
      因此主结论取各核 last L 的**最小值**（提前停下），最大值仅作对照。
      单核或各核 last L 相同时，min==max，与旧行为一致。
    """
    work_path = _default_work_path(args)
    printf_dir = args.printf_dir
    if not printf_dir:
        if not work_path:
            logger.info("错误：未指定工作目录。请传 --log-dir / --printf-dir，或先 export ASCEND_WORK_PATH。")
            return 1
        printf_dir = _find_latest_printf_dir(work_path)
    if not printf_dir:
        where = f"{work_path}/printf/" if work_path else "$ASCEND_WORK_PATH/printf/"
        logger.info(f"错误：在 {where} 下未找到落盘目录。")
        logger.info("      请确认复现时设置了 export ASCEND_WORK_PATH=<dir>（推荐与 --log-dir 相同），")
        logger.info("      且已 export PYPTO_ENABLE_AICORE_PRINT=true 并成功跑过插桩后的用例。")
        return 1

    bins = sorted(glob.glob(os.path.join(printf_dir, "asc_kernel_data_*.bin")))
    if not bins:
        logger.info(f"错误：目录内无 asc_kernel_data_*.bin: {printf_dir}")
        return 1

    tag = args.tag
    logger.info(f"分析 printf 落盘: {printf_dir}")
    logger.info(f"  bin 数={len(bins)}，过滤 tag={tag!r}")
    logger.info("  （终端 aicore error 刷屏可忽略；PROBE 以本目录为准）")
    logger.info("")

    # bin 名通常带 aiv/aic + core 编号：取最后一条，并统计命中数（过滤几乎未进窗口的核）
    per_bin = {}  # name -> {"last": hit, "n": int}
    total_hits = 0
    for bp in bins:
        hits = list(_iter_probe_hits_from_bin(bp, tag))
        total_hits += len(hits)
        if hits:
            per_bin[os.path.basename(bp)] = {"last": hits[-1], "n": len(hits)}

    if not per_bin:
        logger.info(f"错误：未在 bin 中找到 [PROBE:{tag}]。")
        logger.info("      可能 tag 不匹配，或本次运行 printf 未写入（检查 PYPTO_ENABLE_AICORE_PRINT）。")
        return 1

    # 排除几乎没进打点窗口的核（命中过少），避免把「从未跑到」误当成「最早挂住」
    hit_counts = sorted(v["n"] for v in per_bin.values())
    median_n = hit_counts[len(hit_counts) // 2]
    min_active_n = max(10, median_n // 20)  # 至少 10，或中位数的 5%
    active = {k: v for k, v in per_bin.items() if v["n"] >= min_active_n}
    if not active:
        active = per_bin

    ranked = sorted(active.items(), key=lambda kv: (kv[1]["last"]["line"], kv[0]))
    logger.info(
        f"共解析到 {total_hits} 条 PROBE，{len(per_bin)} 个核/bin 有输出"
        f"（活跃核 {len(active)}，命中阈值>={min_active_n}）："
    )

    # last L 分布
    by_last = {}
    for name, meta in ranked:
        last_l = meta["last"]["line"]
        by_last.setdefault(last_l, []).append(name)
    logger.info("  last L 分布（活跃核）：")
    for last_l in sorted(by_last):
        names = by_last[last_l]
        preview = ", ".join(names[:6]) + ("..." if len(names) > 6 else "")
        logger.info(f"    L={last_l}: {len(names)} 核  [{preview}]")

    logger.info("  各核 last（按 L 升序，提前停下的在前）：")
    for name, meta in ranked:
        hit = meta["last"]
        logger.info(f"  [{name}] last L={hit['line']} n={meta['n']} | {hit['snippet'][:100]}")

    early_name, early_meta = ranked[0]
    early = early_meta["last"]
    far_name, far_meta = ranked[-1]
    far = far_meta["last"]
    early_peers = by_last.get(early["line"], [])

    logger.info("")
    logger.info("=" * 78)
    logger.info("自动闭环结论（取活跃核 last L= 的最小值 = 提前停下）")
    logger.info("=" * 78)
    logger.info(f"  来源 bin : {early_name}" + (f" 等共 {len(early_peers)} 核" if len(early_peers) > 1 else ""))
    if len(early_peers) > 1:
        logger.info(f"  同 last 核: {', '.join(early_peers)}")
    logger.info(f"  源文件  : {early['file']}")
    logger.info(f"  最后到达 : L={early['line']}  （插桩前 / .bak 行号）")
    logger.info(f"  源码摘要 : {early['snippet']}")
    if far["line"] != early["line"]:
        logger.info("")
        logger.info(
            f"  对照（最远 last L={far['line']} @ {far_name}）：多为已跑完该窗口的健康核，不宜作挂点。"
        )
    logger.info("")
    logger.info("  判读：优先看提前停下的核；挂点通常在「最后到达行」与「下一个未打印的打点行」之间；")
    logger.info("        若最后一条本身就是非法地址/异常指令，挂点就在该行。")
    logger.info("        有 DFX core id 时，请再核对对应 aic_/aiv_<id>.bin 的 last L。")
    logger.info("=" * 78)
    return 0


def _print_next_steps(args, probe_path: str, tag: str, mode: str = "cce", outer_path: str = None):
    work_path = _default_work_path(args)
    # 未指定时只给占位符，避免暗示某个固定目录名
    work_disp = work_path or "<ASCEND_WORK_PATH>"
    self_path = os.path.abspath(__file__)
    logger.info("=" * 78)
    if mode == "framework":
        logger.info("下一步：框架头已改，需重编入口 TU 后再跑用例")
    else:
        logger.info("下一步：请重新执行用例（不要删除 kernel_aicore 目录，否则插桩被覆盖）")
    logger.info("=" * 78)
    if mode == "framework":
        logger.info("  # 1) 删除入口桩目标文件，迫使 bisheng 带着改过的头重新编译")
        stub_dir = None
        if outer_path and os.path.isfile(outer_path):
            stub_dir = os.path.dirname(outer_path)
        elif outer_path and os.path.isdir(os.path.dirname(outer_path)):
            stub_dir = os.path.dirname(outer_path)
        if stub_dir:
            logger.info(f"  rm -f {stub_dir}/aicore_*.o")
            logger.info(f"  # 入口 CCE 示例: {outer_path}")
        else:
            logger.info("  rm -f <kernel_aicore>/aicore_*.o")
        logger.info("  # 不要删 aicore_*.cpp / 子函数 CCE；只要 .o 失效即可")
        logger.info("")
    logger.info("  export PYPTO_ENABLE_AICORE_PRINT=true      # 打开 PYPTO_AICORE_PRINTF")
    logger.info(f"  export ASCEND_WORK_PATH={work_disp}       # printf 落盘到 $ASCEND_WORK_PATH/printf/")
    logger.info(f"  export ASCEND_PROCESS_LOG_PATH={work_disp}")
    logger.info("  export ASCEND_GLOBAL_LOG_LEVEL=1")
    if args.test_cmd:
        logger.info(f"  {args.test_cmd} >{work_disp}/case_run.log 2>&1")
    else:
        logger.info(f"  <你的用例命令> >{work_disp}/case_run.log 2>&1")
    logger.info("")
    logger.info("执行完用落盘自动闭环（不要靠终端 PROBE）：")
    logger.info(f"  python3 {self_path} --analyze --log-dir {work_disp} --tag {tag}")
    logger.info("")
    logger.info("还原插桩（只恢复文件，无需 --line）：")
    if mode == "framework":
        logger.info(f"  python3 {self_path} --restore --header {probe_path}")
    else:
        logger.info(f"  python3 {self_path} --restore --cce {probe_path}")
    logger.info("=" * 78)


def do_restore(args) -> int:
    path = None
    if args.header:
        path = args.header
    elif args.cce:
        path = args.cce
    else:
        try:
            # 与插桩相同分流：框架模式还原 innerSrc 头，否则还原 outer CCE
            fake = argparse.Namespace(
                header=None,
                cce=None,
                line=None,
                mode=getattr(args, "mode", "auto"),
                info=args.info,
                report_dir=args.report_dir,
            )
            target = _resolve_probe_target(fake)
            path = target["path"]
            logger.info(f"按 mode={target['mode']} 还原: {path}")
        except Exception as e:
            logger.info(f"错误：无法确定要还原的文件: {e}")
            return 1
    path = os.path.abspath(path)
    bak = path + ".bak"
    if not os.path.exists(bak):
        logger.info(f"错误：备份不存在，无需还原: {bak}")
        return 1
    shutil.copy2(bak, path)
    logger.info(f"已还原: {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="debug_aicore_probe: CCE / machine AICore 行级打点与落盘判读（--probe / --analyze / --restore）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--probe",
        action="store_true",
        help="插桩：在 CCE 或框架头目标行窗口插入 PYPTO_AICORE_PRINTF",
    )
    action.add_argument(
        "--analyze",
        action="store_true",
        help="判读：从 printf 落盘 bin 解析 [PROBE] 并给出各核最后到达行",
    )
    action.add_argument(
        "--restore",
        action="store_true",
        help="还原：从同路径 .bak 恢复插桩前文件",
    )

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--report-dir", help="debug_aicore_error.py 的 -out 目录（自动找最新的 info.txt）")
    src.add_argument("--info", help="直接指定 info.txt 路径")
    ap.add_argument("--cce", help="手工指定 CCE 文件（配合 --line；CCE 模式）")
    ap.add_argument("--header", help="手工指定框架头（配合 --line；framework 模式）")
    ap.add_argument("--line", type=int, help="手工指定可疑行号")
    ap.add_argument(
        "--mode",
        choices=("auto", "cce", "framework"),
        default="auto",
        help="仅 --probe/--restore：auto=按 outer/inner 分流；cce=只动生成 CCE；framework=只动框架头",
    )
    ap.add_argument("--window", type=int, default=10, help="目标行前后各插桩的行数，默认 10（仅 --probe）")
    ap.add_argument("--tag", default="T1", help="打点标签，用于多轮插桩/判读区分，默认 T1")
    ap.add_argument("--core0-only", action="store_true", help="只在 core 0 打印，减少日志量（仅 --probe）")
    ap.add_argument(
        "--log-dir",
        help="工作目录，须等于复现时的 ASCEND_WORK_PATH（兼作 ASCEND_PROCESS_LOG_PATH）；"
        "省略则用环境变量 $ASCEND_WORK_PATH（必须已设置）",
    )
    ap.add_argument(
        "--printf-dir",
        help="直接指定 printf 落盘目录（通常为 $ASCEND_WORK_PATH/printf/<ts>），仅 --analyze 使用",
    )
    ap.add_argument("--test-cmd", help="仅用于提示信息，回显你的用例命令（仅 --probe）")
    ap.add_argument(
        "--wait",
        type=int,
        default=8,
        help="前置提示后的等待秒数，期间可 Ctrl+C 中止，默认 8（仅 --probe）",
    )
    ap.add_argument("--force", action="store_true", help="outerSrc 指向入口桩时仍强行按 CCE 插桩（仅 --probe）")
    ap.add_argument("--no-patch-env", action="store_true", help="不自动修复已知的本机头文件问题（仅 --probe）")
    args = ap.parse_args()

    if args.probe or args.restore:
        if not args.cce and not args.header and not args.report_dir and not args.info:
            ap.error("--probe / --restore 需要 --report-dir / --info / --cce / --header 之一")

    try:
        if args.restore:
            return do_restore(args)
        if args.analyze:
            return do_analyze(args)

        # --probe：先解析目标再按模式提示（framework 与 CCE 前置条件不同）
        try:
            target = _resolve_probe_target(args)
        except RuntimeError as e:
            logger.info(f"错误：{e}")
            return 1

        _precheck_pause(args.wait, mode=target.get("mode") or "cce")

        if not env_selfcheck(auto_patch=not args.no_patch_env):
            logger.info("")
            logger.info("警告：CCE print 通道未就绪，插桩后重跑大概率编译失败。")
            logger.info("      可加 --no-patch-env 自行处理，或检查 pypto 安装是否完整。")

        return do_probe(args, resolved_target=target)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("已中止。")
        return 130
    except Exception as e:
        logger.info(f"错误：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
