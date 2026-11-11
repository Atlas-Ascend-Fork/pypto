#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Whole-model run for the nested-control-flow demo: pick a branch and dispatch the model.

The args -> run mapping stays visible here, one demo at a time: cpu (each op's torch_defn), npu (real
device kernels), or cpu + ``--soc_version`` (the env-level NPU simulator).

``--branch`` selects which conditional is taken, and therefore which ONE op runs. The model is used
unscripted here, so Python evaluates the conditional directly. One op per process also suits the
env-level simulator, which initializes and finalizes the runtime per kernel launch.

Nothing may import this file: it runs as ``__main__``.
"""
import argparse
import logging
import pathlib as _pl
import sys as _sys

import torch

import pypto

# The exported_custom_op_litenpu helper library lives under the repo-root tools/ directory.
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))  # <repo root>/tools
from exported_custom_op_litenpu.run.demo_run import (
    RunCliError,
    RunEnvError,
    add_run_args,
    enable_eslmodel,
    move_inputs_to_npu,
    npu_device_id,
    print_outputs,
    pypto_run_context,
    scenario_label,
    validate_args,
)
from model import SHAPE, CustomModel

# Console output for this demo: its own stdout handler at INFO, no root propagation.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _console = logging.StreamHandler(_sys.stdout)
    _console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_console)
logger.setLevel(logging.INFO)
logger.propagate = False

# --branch -> the (use_add, use_sub) flags that select that branch in model.CustomModel.forward.
_BRANCH_FLAGS = {"add": (True, False), "sub": (False, True), "mul": (False, False)}


def run_demo(model, args):
    input_data0 = torch.rand(SHAPE, dtype=torch.float16)
    input_data1 = torch.rand(SHAPE, dtype=torch.float16)
    use_add, use_sub = _BRANCH_FLAGS[args.branch]
    inputs = (input_data0, input_data1, torch.tensor(use_add), torch.tensor(use_sub))

    validate_args(args)
    logger.info("run scenario: %s, branch: %s", scenario_label(args), args.branch)

    if args.device == "npu":
        device_id = npu_device_id()
        torch.npu.set_device(device_id)
        inputs = move_inputs_to_npu(inputs)
        with pypto_run_context(run_mode=pypto.RunMode.NPU):
            out = model(*inputs)
    elif args.soc_version:
        with enable_eslmodel(), pypto_run_context(run_mode=pypto.RunMode.SIM, soc_version=args.soc_version):
            out = model(*inputs)
    else:
        out = model(*inputs)

    print_outputs(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    parser.add_argument("--branch", choices=sorted(_BRANCH_FLAGS), default="add",
                        help="which conditional branch to take, and so which single op runs")
    args = parser.parse_args()
    try:
        run_demo(CustomModel(), args)
    except RunCliError as e:
        parser.error(str(e))
    except RunEnvError as e:
        _sys.exit(f"run error: {e}")  # clean exit (e.g. no NPU to auto-detect a soc), not a traceback
