# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import glob
import logging
import os

import numpy as np
import torch

import pypto

logger = logging.getLogger(__name__)


def compare_cos(input_1, input_2):
    input_1 = input_1.reshape(-1).astype(np.float64)
    input_2 = input_2.reshape(-1).astype(np.float64)
    logger.info("max diff: %s", np.max(np.abs(input_1 - input_2)))
    index = np.argmax(np.abs(input_1 - input_2))
    logger.info("max diff index = %s, intput_1 value: %s, intput_2 value: %s", index, input_1[index], input_2[index])
    logger.info("average diff: %s", np.mean(np.abs(input_1 - input_2)))
    ab = np.sum(input_1 * input_2)
    aa = np.sqrt(np.sum(input_1 * input_1))
    bb = np.sqrt(np.sum(input_2 * input_2))
    if aa == 0 and bb == 0:
        cos = 1.0
    elif aa == 0 or bb == 0:
        cos = 0.0
    else:
        cos = ab / (aa * bb)
    logger.info("cosine similarity: %s", cos)
    return cos


def generated_cpp_contains(keyword, expect_absent=False):
    """Return True if the generated kernel cpp under LogTopFolder()/kernel_aicore contains ``keyword``.

    Logging is expectation-aware: by default a missing keyword is logged as a
    warning; when ``expect_absent`` is True, an unexpected match is logged as a
    warning instead.
    """
    kernel_dir = os.path.join(pypto.pypto_impl.LogTopFolder(), "kernel_aicore")
    logger.info("generated_cpp_contains: searching '%s' under %s", keyword, kernel_dir)
    if not os.path.isdir(kernel_dir):
        logger.warning("generated_cpp_contains: kernel_aicore dir does not exist: %s", kernel_dir)
        return False
    for path in glob.glob(os.path.join(kernel_dir, "*.cpp")):
        with open(path, encoding="utf-8", errors="replace") as handle:
            if keyword in handle.read():
                if expect_absent:
                    logger.warning(
                        "generated_cpp_contains: unexpectedly found '%s' in %s", keyword, os.path.basename(path)
                    )
                else:
                    logger.info("generated_cpp_contains: found '%s' in %s", keyword, os.path.basename(path))
                return True
    if expect_absent:
        logger.info("generated_cpp_contains: '%s' not found (as expected)", keyword)
    else:
        logger.warning("generated_cpp_contains: '%s' not found in any generated cpp", keyword)
    return False


def check_atol_rtol(input_1, input_2, rtol, atol, name="", scale_tol=1e-2):
    """Elementwise atol/rtol check plus a global magnitude-ratio check.

    Pass criterion:
      - |actual - golden| <= atol + rtol * |golden| for every element;
      - ||actual|| / ||golden|| ~= 1 (catches a pure global scaling, which
        compare_cos cannot detect because cosine is scale-invariant).
    """
    if isinstance(input_1, torch.Tensor):
        input_1 = input_1.detach().cpu().numpy()
    if isinstance(input_2, torch.Tensor):
        input_2 = input_2.detach().cpu().numpy()
    a = np.asarray(input_1, dtype=np.float64).reshape(-1)
    g = np.asarray(input_2, dtype=np.float64).reshape(-1)

    diff = np.abs(a - g)
    tol = atol + rtol * np.abs(g)
    bad = diff > tol
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    g_norm = float(np.linalg.norm(g))
    a_norm = float(np.linalg.norm(a))
    if g_norm == 0.0:
        ratio = 1.0 if a_norm == 0.0 else 0.0
    else:
        ratio = a_norm / g_norm
    scale_ok = abs(ratio - 1.0) <= scale_tol

    prefix = f"{name}: " if name else ""
    logger.info(
        "%smax diff: %s, mean diff: %s, scale ratio: %s",
        prefix,
        max_diff,
        mean_diff,
        ratio,
    )
    if bad.any() or not scale_ok:
        idx = int(np.argmax(diff)) if diff.size else 0
        raise AssertionError(
            f"{prefix}precision check failed (rtol={rtol}, atol={atol}): "
            f"bad={int(bad.sum())}/{a.size}, max_diff={max_diff:.6g} @ idx={idx} "
            f"(actual={a[idx]:.6g}, golden={g[idx]:.6g}), mean_diff={mean_diff:.6g}, "
            f"scale_ratio={ratio:.6g}, scale_ok={scale_ok}"
        )
    return {"max_diff": max_diff, "mean_diff": mean_diff, "scale_ratio": ratio}


def check_nan(tensor, name=""):
    """Raise AssertionError if tensor contains any NaN values."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        prefix = f"{name}: " if name else ""
        raise AssertionError(f"{prefix}tensor contains NaN values")


def load_bin(filepath, dtype=np.float16):
    with open(filepath, 'rb') as f:
        content = f.read()
    data = np.frombuffer(content, dtype=dtype)
    return data


def save_bin(output, filepath):
    filepath = filepath.replace("/", "_")
    with open(filepath, 'wb') as f:
        f.write(output.tobytes())
        f.close()
    return
