#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side list-tensor descriptor builder for runtime-variadic (DYNAMIC) inputs.

A kernel reads a DYNAMIC input through the list-tensor descriptor the framework passes
(see ``pl.dynamic_input_ptr`` / ``pl.dynamic_input_dim``). When the kernel is launched
directly (``kernel[stream, grid](...)``) the caller synthesizes that descriptor on the
host; ``make_list_tensor_desc`` builds it in the layout the device side decodes -- the
same one AscendC ``ListTensorDesc`` reads:

- dim == 0 (no shape info): ``[24, (count << 32), 0xFFFFFFFF, addr_0 .. addr_{N-1}]``
- dim != 0 (each entry carries a shape): ``[offset, (count << 32) | dim, N shape structs
  of (1 + dim) words, addr_0 .. addr_{N-1}]`` where ``offset = 8 + N * (1 + dim) * 8``

The dim != 0 form stores one shape struct per tensor, so axis lengths may differ per
tensor; the dimensionality itself is global (one dim word, one uniform struct stride),
so every listed tensor must share the same dim.
"""

from __future__ import annotations

__all__ = ["make_list_tensor_desc"]


from typing import Sequence

import torch

from pypto_pro._errors import InvalidArgument, InvalidShape


def make_list_tensor_desc(
    addrs: Sequence[int],
    shape: Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Build the list-tensor descriptor for a runtime-variadic (DYNAMIC) input.

    Returns an int64 tensor holding the descriptor consumed by ``pl.dynamic_input_ptr``
    and ``pl.dynamic_input_dim``; it is created on CPU, so move it to the launch device
    (``desc.to(device)``) before launching the kernel.

    Args:
        addrs: Device address of each listed tensor, e.g. ``[t.data_ptr() for t in tensors]``.
        shape: Shape carried by the listed tensors. ``None`` (default) builds the
            dim==0 form that carries addresses only. A flat sequence of positive ints
            is one shape shared by every listed tensor; a sequence of such sequences
            carries each tensor's shape individually (its length must match ``addrs``).
            All tensors must share the same dimensionality -- the descriptor holds a
            single global dim and a uniform struct stride.
    """
    addrs = list(addrs)
    if not addrs:
        raise InvalidArgument("make_list_tensor_desc requires at least one tensor address")
    count = len(addrs)

    if shape is None:
        words = [24, count << 32, 0xFFFFFFFF, *addrs]
        return torch.tensor(words, dtype=torch.int64)

    shape = list(shape)
    if shape and all(isinstance(s, (list, tuple)) for s in shape):
        if len(shape) != count:
            raise InvalidArgument(
                f"make_list_tensor_desc shape holds {len(shape)} per-tensor axis lists "
                f"but there are {count} tensor addresses; the two counts must match"
            )
        per_tensor = [list(s) for s in shape]
        per_tensor_mode = True
    else:
        per_tensor = [shape] * count
        per_tensor_mode = False

    dim = len(per_tensor[0])
    for i, shape_i in enumerate(per_tensor):
        if len(shape_i) != dim:
            raise InvalidShape(
                f"make_list_tensor_desc shape[{i}] has {len(shape_i)} axes but tensor 0 has {dim}; "
                "all listed tensors must share the same dimensionality (one global dim, one struct stride)"
            )
        prefix = f"shape[{i}]" if per_tensor_mode else "shape"
        for axis, dim_len in enumerate(shape_i):
            if isinstance(dim_len, bool) or not isinstance(dim_len, int) or dim_len <= 0:
                raise InvalidShape(
                    f"make_list_tensor_desc {prefix}[{axis}] must be a positive integer, got {dim_len!r}"
                )
    if dim == 0:
        raise InvalidShape(
            "make_list_tensor_desc shape must have at least one axis; omit shape for the addresses-only (dim=0) form"
        )

    struct_size = 1 + dim
    data_ptr_offset = 8 + count * struct_size * 8
    words = [0] * (1 + count * struct_size + count)
    words[0] = data_ptr_offset
    words[1] = dim | (count << 32)  # struct 0 meta: low32 dim, high32 count; later structs stay 0
    for i in range(count):
        base = 1 + i * struct_size
        for d, dim_len in enumerate(per_tensor[i]):
            words[base + 1 + d] = dim_len
    addr_off = data_ptr_offset // 8
    for i, addr in enumerate(addrs):
        words[addr_off + i] = addr
    return torch.tensor(words, dtype=torch.int64)
