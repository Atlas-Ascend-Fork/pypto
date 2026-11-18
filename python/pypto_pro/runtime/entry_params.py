# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND OF EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared launch entry-parameter derivation for the AICore launcher and the AICPU glue.

Both the AICore fast-launch signature (``jit``) and the AICPU glue (``aicpu_compile``)
derive their C entry parameters from the same ``ParamSpec`` metadata; this module owns
that single derivation so the two ABI views cannot drift. It also owns
:func:`pack_launch_args`: validating one launch's runtime arguments against the plan
and flattening them into the ABI list (tensor pointers, scalars, the dynamic-shape tail).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypto_pro import DataType

from .._errors import InvalidOperation, InvalidShape, InvalidType

if TYPE_CHECKING:
    # Annotation-only: the plan/spec types stay jit-owned (no runtime import cycle).
    from .jit import ParamSpec, _LaunchPlan


def cpp_type_from_dtype(dtype, *, param_name: str) -> str:
    """Return the CCE C++ scalar/element type string for a DataType."""
    if dtype is None:
        raise InvalidType(f"Parameter '{param_name}' has no dtype")
    cpp_type = dtype.to_c_type_string()
    if not cpp_type or cpp_type == "unknown":
        raise InvalidType(f"Unsupported CCE ABI dtype {dtype} for parameter '{param_name}'")
    return cpp_type


def iter_tensor_dynamic_dims(spec):
    """Yield dynamic dimension symbols from a tensor param spec."""
    if spec.kind.name != "TENSOR" or spec.shape is None:
        return
    for dim in spec.shape:
        if isinstance(dim, str):
            yield dim


def collect_dyn_vars(param_specs) -> list[str]:
    """Return dynamic shape variable names in first-occurrence order (mirrors ptoas codegen)."""
    seen: set[str] = set()
    result: list[str] = []
    for spec in param_specs:
        for dim in iter_tensor_dynamic_dims(spec):
            if dim not in seen:
                result.append(dim)
                seen.add(dim)
    return result


def entry_params_from_param_specs(param_specs) -> list[tuple[str, str, bool]]:
    """Build launcher entry params from IR metadata instead of parsing generated C++.

    Returns ``(cpp_type, name, is_ptr)`` triples: the AICore launcher consumes them
    directly; the AICPU glue folds ``is_ptr`` back into ``type*`` field strings.
    """
    params: list[tuple[str, str, bool]] = []
    for spec in param_specs:
        kind = spec.kind.name
        if kind in ("TENSOR", "PTR"):
            params.append((cpp_type_from_dtype(spec.dtype, param_name=spec.name), spec.name, True))
        elif kind in ("TILING", "WINDOW"):
            # Both cross the ABI as one raw GM byte pointer.
            params.append(("uint8_t", spec.name, True))
        elif kind == "SCALAR":
            params.append((cpp_type_from_dtype(spec.dtype, param_name=spec.name), spec.name, False))
    params.extend(("int64_t", dim_name, False) for dim_name in collect_dyn_vars(param_specs))
    return params


def _validate_scalar_arg(i: int, arg, spec: ParamSpec) -> None:
    """Validate one scalar arg against its ParamSpec."""
    if not isinstance(arg, (int, float, bool)):
        raise InvalidType(f"arg[{i}] '{spec.name}': expected Python scalar (int/float/bool), got {type(arg).__name__}")
    # bool is a subclass of int: valid for BOOL and integer dtypes only.
    if isinstance(arg, bool):
        if not (spec.dtype == DataType.BOOL or spec.dtype.is_signed_int() or spec.dtype.is_unsigned_int()):
            raise InvalidType(
                f"arg[{i}] '{spec.name}': bool value passed for non-boolean/non-integer dtype {spec.dtype}"
            )
    elif isinstance(arg, float) and not spec.dtype.is_float():
        raise InvalidType(f"arg[{i}] '{spec.name}': float value passed for non-float dtype {spec.dtype}")
    elif isinstance(arg, int) and not spec.dtype.is_signed_int() and not spec.dtype.is_unsigned_int():
        raise InvalidType(f"arg[{i}] '{spec.name}': int value passed for non-integer dtype {spec.dtype}")


def _pack_tiling_arg(args: tuple) -> tuple:
    """If the last arg is a tiling class instance, replace it with a device buffer tensor.

    The tiling instance is serialized to its C-struct byte layout, wrapped in a uint8
    CPU tensor, and moved to the same device as the other tensor args. The kernel
    receives a single ``uint8_t*`` pointer and copies the struct in at its entry.

    The returned tuple holds the device tensor as its last element, which keeps it alive
    through the launch (the caller's ``args`` stays referenced until ``call_kernel``).

    Args:
        args: Runtime arguments tuple passed to launch.

    Returns:
        args with the tiling instance replaced by the device tensor (unchanged if no
        tiling instance is present).
    """
    import torch

    from pypto_pro.language.typing._tiling import is_tiling_class, tiling_instance_to_bytes

    if not args:
        return args
    last_arg = args[-1]
    if not is_tiling_class(type(last_arg)):
        return args

    buf = tiling_instance_to_bytes(last_arg)
    # Pick the device from an existing tensor arg (inputs/outputs are already on device).
    device = None
    for a in args[:-1]:
        dev = getattr(a, "device", None)
        if dev is not None:
            device = dev
            break
    cpu_t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    tiling_t = cpu_t.to(device) if device is not None else cpu_t
    return args[:-1] + (tiling_t,)


def pack_launch_args(plan: "_LaunchPlan", args: tuple, window_cls=None) -> list:
    """Validate one launch's arguments and return the flat ABI list.

    Shared by the device launcher and the AICPU host shim: the slots order is
    the IR parameter order, followed by the dynamic-shape
    tail in first-occurrence order. ``window_cls`` is the pre-resolved pybind
    Window class of the device launch path; window params cannot appear on an
    AICPU kernel, whose shim passes the default None.
    """
    # torch is resolved per call: the UT harness swaps sys.modules["torch"] for a
    # mock after this shared module may already be imported.
    import torch

    if plan.has_tiling:
        args = _pack_tiling_arg(args)
    if len(args) != plan.n_params:
        raise InvalidType(f"Expected {plan.n_params} args, got {len(args)}")

    abi_args = []
    append = abi_args.append
    dyn_values: list = []
    for slot in plan.slots:
        index = slot[1]
        arg = args[index]
        if not slot[0]:
            _validate_scalar_arg(index, arg, slot[2])
            append(arg)
            continue
        if index in plan.window_indices:
            # pld.Window: the ABI value is the device descriptor address itself.
            if window_cls is None or not isinstance(arg, window_cls):
                raise InvalidType(
                    f"arg[{index}] '{slot[2]}': expected a pld.Window, got {type(arg).__name__}"
                )
            if not arg.connected():
                raise InvalidOperation(
                    f"arg[{index}] '{slot[2]}': window is not connected; call win.connect() before launching"
                )
            append(arg.data_ptr())
            continue
        _, _, name, expected_dtypes, rank, consts, dyn_dims, pack = slot
        if arg is None:
            append(None)  # ctypes maps None to a null void*
            if dyn_dims:
                dyn_values.extend([0] * len(dyn_dims))
            continue
        if not isinstance(arg, torch.Tensor):
            raise InvalidType(f"arg[{index}] '{name}': expected torch.Tensor, got {type(arg).__name__}")
        if rank >= 0:
            if expected_dtypes is not None and arg.dtype not in expected_dtypes:
                expected = " or ".join(str(dtype) for dtype in expected_dtypes)
                raise InvalidType(
                    f"arg[{index}] '{name}': dtype mismatch — expected {expected}, got {arg.dtype}"
                )
            # One .shape per tensor: reused for the positivity check, the rank check,
            # the constant-dimension checks and this parameter's slice of the dynamic
            # ABI tail.
            shape = arg.shape
            if len(shape) != rank:
                raise InvalidType(
                    f"arg[{index}] '{name}': rank mismatch — expected {rank}D, got {len(shape)}D"
                )
            # Sub-byte dtype: torch reported the packed uint8 storage, so scale the
            # innermost axis back to elements (see _pack_factor). Packing does not
            # change the rank, so the check above reads either shape the same way.
            if pack != 1 and rank:
                shape = tuple(shape[:-1]) + (shape[-1] * pack,)
            # The compile path rejects a non-positive runtime dimension inside the
            # shape-policy bind (_validate_positive_int). A cache hit skips that bind
            # entirely, so without this a zero-sized tensor would raise on the launch
            # that compiled the kernel and be accepted by every launch after it. One
            # C-level pass over the torch.Size already in hand; `rank` guards the
            # 0-d case, where min() would see an empty sequence.
            if rank and min(shape) <= 0:
                raise InvalidShape(
                    f"arg[{index}] '{name}': runtime dimensions must be positive, got {tuple(shape)}"
                )
            for dim, expected in consts:
                if shape[dim] != expected:
                    raise InvalidType(
                        f"arg[{index}] '{name}': dim[{dim}] mismatch — "
                        f"expected {expected}, got {shape[dim]}"
                    )
            for dim in dyn_dims:
                dyn_values.append(shape[dim])
        append(arg.data_ptr())

    for var_name, where in plan.dyn_checks:
        first = -1
        for param_index, dim, pack_mult in where:
            other = args[param_index]
            if other is None:
                continue
            # pack_mult is 1 unless this is a sub-byte parameter's packed axis.
            actual = other.shape[dim] * pack_mult
            if first < 0:
                first = actual
            elif actual != first:
                raise InvalidShape(
                    f"arg[{param_index}] '{plan.slots[param_index][2]}': dynamic shape variable "
                    f"'{var_name}' mismatch — previously {first}, got {actual} at dim[{dim}]"
                )
    abi_args.extend(dyn_values)
    return abi_args
