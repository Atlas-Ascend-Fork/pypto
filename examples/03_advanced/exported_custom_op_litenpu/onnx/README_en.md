# onnx/

Runnable ONNX export demos. Each demo is a FOLDER `<name>/` holding five files: `kernel.py` (the kernel
body), `op.py` (inference + CPU reference + the `ExportedCustomOp(...)` declaration), `model.py` (the
`nn.Module` + shape constants), `run_demo.py` (the RUN entry point) and `export_demo.py` (the EXPORT
entry point). The import chain forks one-way, `kernel -> op -> model -> {export_demo, run_demo}`; nothing
may import either entry point, since each runs as `__main__` and would declare the op twice if imported.
`export_demo.py` exports the model to a `.onnx` custom-op artifact, and
`tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py` builds one `libcust_opapi.so` from that artifact and places it under `framework/onnx/` for ATC/GE discovery.

You do not write the torch op or the ONNX node yourself: you declare them as config on
`ExportedCustomOp` (`torch_op_qualname=` + `onnx_spec=OnnxSymbolicSpec(...)`) in `op.py` and pypto
generates them automatically during export. `torch_defn` (the CPU reference `run_demo.py` uses) is
declared there too.

## Dependencies

The export path in `export_demo.py` depends on `onnx`, an **optional dependency**: only this ONNX export
demo needs it. pypto core and the CPU/device run path of `run_demo.py` do not
(`pypto.extensions.torch_custom_op_litenpu` imports `onnx` lazily inside functions). Install it before the
first export:

```bash
pip install onnx
```

Without it, `export_demo.py` raises `ModuleNotFoundError: No module named 'onnx'`.

## Version requirement

This demo depends on `pypto.extensions.torch_custom_op_litenpu` (providing `ExportedCustomOp`,
`common/run.py`, and more), which currently ships only on the PyPTO `master` source; the PyPTO released with
CANN 9.2.0 does not contain these modules. Before running, confirm you are using a PyPTO built and installed
from this repository's source, not the PyPTO installed with CANN:

```bash
python3 -c "import pypto; print(pypto.__file__)"
```

If the path points into a CANN install (e.g. `.../Ascend/.../site-packages/pypto`), the bundled older copy
is loaded: build and install PyPTO from this repository's source following
[Build and install](../../../../docs/en/install/build_and_install.md), then rerun. Errors like
`ModuleNotFoundError: No module named 'pypto.extensions...'` or
`AttributeError: ... has no attribute 'ExportedCustomOp'` also mean a version mismatch — reinstall as above
rather than treating them as a demo or source defect.

Run from this directory (`examples/03_advanced/exported_custom_op_litenpu/onnx/`):

```bash
python3 add/export_demo.py /tmp/add.onnx        # export only, never runs
python3 ../../../../tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py /tmp/add.onnx
python3 add/run_demo.py                         # run on cpu (each op's torch_defn), never exports
python3 add/run_demo.py --device=npu            # run the real kernels on a device
python3 add/run_demo.py --soc_version=Kirin9030 # run on the env-level NPU simulator
```

`run_demo.py` accepts `--device={cpu,npu}` and `--soc_version=<soc>`; `--soc_version` selects the
env-level NPU simulator and pairs with `--device=cpu` or with `--device` omitted, never with
`--device=npu`.

## Authoring rules

A DIRECT `@pypto.frontend.jit` kernel SHOULD pin `runtime_options={"run_mode": pypto.RunMode.SIM}` on its
decorator. The deployed `.so` is forced to SIM regardless, but the frontend defaults run_mode per-box at
import, so pinning SIM avoids an import-time "NPU is not available" surprise on a CANN host.

Each single-pypto-op demo has ONE *stem* = `<compute>[_<attr>...]`, the compute first, then the
attributes that distinguish this demo from the plain one. Every identifier derived from the demo uses
that stem: the folder `<stem>` (no `_kernel` suffix), the kernel fn `<stem>_kernel` (or
`create_<stem>_kernel` for a factory), the inference fns `<stem>_infer_shape` / `<stem>_infer_dtype`, the
CPU reference `<stem>_torch`, and the torch registry key `pypto::<stem>`.

## op_type = the COMPUTE · torch_op_qualname = the UNIQUE per-op key

`op_type` (the `OnnxSymbolicSpec(op_type=...)` string) names the COMPUTE and becomes the graph node type
a consumer sees (`"Add"` -> `PyptoCustomOpAdd`), so it MAY repeat across demos that compute the same
thing. `torch_op_qualname` (`"pypto::add"`) is the torch-registry key and carries the feature stem; it
too repeats across demos exposing the same op, and the torch registry is process-global, which is why
each demo must run in its own process.
