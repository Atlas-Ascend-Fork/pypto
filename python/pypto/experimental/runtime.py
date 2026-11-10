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
"""Experimental runtime options.

Use set_runtime_options / get_runtime_options for runtime features that are not
yet stable enough for pypto.frontend.jit(runtime_options=...).
"""

from typing import Dict, List, Optional, Union

from ..config import get_current_scope, set_options

# Experimental runtime keys exposed by get_runtime_options. Stable runtime options such as
# stitch_function_max_num stay on @jit(runtime_options=...) and are not returned here.
# Register a new experimental option here when it is added to set_runtime_options.
_EXPERIMENTAL_RUNTIME_OPTIONS = (
    "stitch_function_num_per_pool",
)


def _validate_stitch_function_num_per_pool(value) -> List[int]:
    """Validate the runtime option stitch_function_num_per_pool.

    Expects a list of 3 ints in [0, 1024], one per workspace pool
    (root_inner, assemble_outcast, exclusive_outcast).
    """
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            f"Invalid stitch_function_num_per_pool: '{value}'. Expected a list of 3 ints in "
            f"[0, 1024], meaning [root_inner_depth, assemble_outcast_depth, exclusive_outcast_depth]."
        )
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 1024 for x in value):
        raise ValueError(
            f"Invalid stitch_function_num_per_pool: '{value}'. Each element must be an int in [0, 1024]."
        )
    return list(value)


def set_runtime_options(
    *,
    stitch_function_num_per_pool: Optional[List[int]] = None,
):
    """
    Set experimental runtime options.

    Parameters
    ---------
    stitch_function_num_per_pool : list of int
        Experimental. Controls the stitch depth of the three Workspace memory pools,
        format [root_inner_depth, assemble_outcast_depth, exclusive_outcast_depth].
        Defaults to [0, 0, 0], which disables the precise Workspace mode; any non-zero
        element enables it. Each element is an int in [0, 1024]. This is an experimental
        feature that may change or be removed in future releases.
    """

    if stitch_function_num_per_pool is not None:
        stitch_function_num_per_pool = _validate_stitch_function_num_per_pool(stitch_function_num_per_pool)
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(runtime_options=options_dict)


def get_runtime_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get experimental runtime options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        Registered experimental runtime options. Currently only stitch_function_num_per_pool.
        Stable @jit runtime options are not included.
    """

    scope = get_current_scope()
    runtime_options = scope.get_runtime_options()
    return {key: runtime_options[key] for key in _EXPERIMENTAL_RUNTIME_OPTIONS if key in runtime_options}
