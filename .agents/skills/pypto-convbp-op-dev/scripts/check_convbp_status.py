#!/usr/bin/env python3
"""Check which dx/dw conv-backward modules already exist in a pypto repo.

Outputs a JSON report with one entry per module: present / missing.
Used to decide fresh vs incremental development before touching code.
"""
import argparse
import json
from pathlib import Path
import sys

DX_MODULES = {
    "tilefwk_op.h ConvBp namespace": "framework/include/tilefwk/tilefwk_op.h",
    "tile_shape.h ConvBpTile": "framework/include/tilefwk/tile_shape.h",
    "opcode OP_CONV_BP_INPUT_2D": "framework/src/interface/operation/opcode.h",
    "opcode.cpp RegisterCube OP_CONV_BP_INPUT_2D": "framework/src/interface/operation/opcode.cpp",
    "op_infer_shape L1CopyInConvBpInferFunc": "framework/src/interface/operation/op_infer_shape_impl.cpp",
    "conv_bp_pto.h tileop header": "framework/src/interface/tileop/conv/conv_bp_pto.h",
    "TileOpImpl.h includes conv_bp_pto": "framework/src/interface/tileop/TileOpImpl.h",
    "convbp dx operation impl": "framework/src/interface/operation/convbp_operation_impl.cpp",
    "codegen_mte_convbp.cpp": "framework/src/codegen/npu/codegen_mte_convbp.cpp",
    "python convbp.py": "python/pypto/op/convbp.py",
    "codegen ut test_codegen_dyn_convbp":
        "framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_mte/test_codegen_dyn_convbp.cpp",
    "python ut test_convbp_op": "python/tests/ut/operation/test_convbp_op.py",
    "python st test_convbp": "python/tests/st/test_convbp.py",
}

DX_TEXT_MARKERS = {
    "tilefwk_op.h ConvBp namespace": "ConvBpTileL1Info",
    "tile_shape.h ConvBpTile": "ConvBpTile",
    "opcode OP_CONV_BP_INPUT_2D": "OP_CONV_BP_INPUT_2D",
    "opcode.cpp RegisterCube OP_CONV_BP_INPUT_2D": "OP_CONV_BP_INPUT_2D",
    "op_infer_shape L1CopyInConvBpInferFunc": "L1CopyInConvBpInferFunc",
    "conv_bp_pto.h tileop header": "TLoad2DDX",
    "TileOpImpl.h includes conv_bp_pto": "conv_bp_pto.h",
    "convbp dx operation impl": "ConvBackwardInput",
    "codegen_mte_convbp.cpp": "GenMemL1ToL0Load2DConvBpDx",
    "python convbp.py": "conv_backward_input",
    "codegen ut test_codegen_dyn_convbp": "ConvBp",
    "python ut test_convbp_op": "conv_backward_input",
    "python st test_convbp": "conv_backward_input",
}

DX_ALT_FILES = {
    "convbp dx operation impl": [
        "framework/src/interface/operation/convbp_operation_impl.cpp",
        "framework/src/interface/operation/convbp_dx_operation_impl.cpp",
    ],
}

DW_MODULES = {
    "tile_shape split ConvBpFilterTile": "framework/include/tilefwk/tile_shape.h",
    "tilefwk_op.h ConvBackwardFilter": "framework/include/tilefwk/tilefwk_op.h",
    "opcode OP_CONV_BP_FILTER_2D": "framework/src/interface/operation/opcode.h",
    "opcode.cpp RegisterCube OP_CONV_BP_FILTER_2D": "framework/src/interface/operation/opcode.cpp",
    "convbp dw operation impl": "framework/src/interface/operation/convbp_dw_operation_impl.cpp",
    "convbp dx impl renamed": "framework/src/interface/operation/convbp_dx_operation_impl.cpp",
    "tileop TLoadConvBPDwDedy": "framework/src/interface/tileop/conv/conv_bp_pto.h",
    "codegen GenMemL1CopyInConvBpDwDy": "framework/src/codegen/npu/codegen_mte_convbp.cpp",
    "infer_tensor_format FZ2ND": "framework/src/passes/tensor_graph_pass/infer_tensor_format.cpp",
    "python conv_backward_filter": "python/pypto/op/convbp.py",
    "bindings filter tile api": "python/src/bindings/controller.cpp",
    "framework ut test_operation_impl_conv": "framework/tests/ut/interface/src/operation/test_operation_impl_conv.cpp",
    "docs conv_backward_filter": "docs/zh/api/tensor_api/operation/pypto-conv_backward_filter.md",
}

DW_TEXT_MARKERS = {
    "tile_shape split ConvBpFilterTile": "ConvBpFilterTile",
    "tilefwk_op.h ConvBackwardFilter": "ConvBackwardFilter",
    "opcode OP_CONV_BP_FILTER_2D": "OP_CONV_BP_FILTER_2D",
    "opcode.cpp RegisterCube OP_CONV_BP_FILTER_2D": "OP_CONV_BP_FILTER_2D",
    "convbp dw operation impl": "ConstructDwTileGraph",
    "convbp dx impl renamed": "ConvBackwardInput",
    "tileop TLoadConvBPDwDedy": "TLoadConvBPDwDedy",
    "codegen GenMemL1CopyInConvBpDwDy": "GenMemL1CopyInConvBpDwDy",
    "infer_tensor_format FZ2ND": "srcFormat == TileOpFormat::TILEOP_FRACTAL_Z",
    "python conv_backward_filter": "conv_backward_filter",
    "bindings filter tile api": "SetConvBpFilterTile",
    "framework ut test_operation_impl_conv": "ConvBackwardFilter",
    "docs conv_backward_filter": "conv_backward_filter",
}


def check_module(pypto_root: Path, name: str, files, marker: str, alt_files=None):
    candidates = alt_files if alt_files else [files]
    for rel in candidates:
        path = pypto_root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return {"status": "error", "file": rel, "detail": str(exc)}
        if marker in text:
            return {"status": "present", "file": rel}
    return {"status": "missing", "file": candidates[0]}


def check_op(pypto_root: Path, op: str):
    if op == "dx":
        modules, markers, alts = DX_MODULES, DX_TEXT_MARKERS, DX_ALT_FILES
    else:
        modules, markers, alts = DW_MODULES, DW_TEXT_MARKERS, {}
    report = {}
    for name, rel in modules.items():
        alt = alts.get(name)
        report[name] = check_module(pypto_root, name, rel, markers[name], alt)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto-root", required=True, help="path to the pypto source repo root")
    parser.add_argument("--op", choices=["dx", "dw", "both"], default="both", help="which operator to check")
    args = parser.parse_args()

    root = Path(args.pypto_root).expanduser().resolve()
    if not root.is_dir():
        print(json.dumps({"error": f"pypto root not found: {root}"}, ensure_ascii=False))
        return 1
    if not (root / "framework").is_dir():
        print(json.dumps({"error": f"not a pypto repo (no framework/ under {root})"}, ensure_ascii=False))
        return 1

    ops = ["dx", "dw"] if args.op == "both" else [args.op]
    result = {"pypto_root": str(root)}
    for op in ops:
        report = check_op(root, op)
        present = sum(1 for v in report.values() if v["status"] == "present")
        result[op] = {
            "summary": {"present": present, "missing": len(report) - present, "total": len(report)},
            "modules": report,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
