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
"""``infer_shape`` hooks authored under postponed evaluation (PEP 563).

The future import is the point of this module: it makes ``inspect.signature`` report every annotation as
its source string, which a test module that evaluates its own annotations cannot reproduce.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import torch

if TYPE_CHECKING:  # a name that resolves nowhere at runtime, for the fallback case below
    from nowhere import Unresolvable


def str_attr(x: torch.Size, layout: str) -> torch.Size:
    return torch.Size((*x, len(layout)))


def int_attr(x: torch.Size, n: int) -> torch.Size:
    return torch.Size((*x, n))


def listint_attr(x: torch.Size, crop: list[int]) -> torch.Size:
    return torch.Size([x[0], crop[0], crop[1]])


def str_attr_with_an_unresolvable_sibling(x: torch.Size, layout: str, mode: Unresolvable = None) -> torch.Size:
    return torch.Size((*x, len(layout)))


def annotated_str_attr(x: torch.Size, layout: Annotated[str, "the memory layout"]) -> torch.Size:
    return torch.Size((*x, len(layout)))


class CallableStrAttr:
    """A callable-object hook whose CLASS annotates ``layout`` as the wrong type for ``__call__``."""
    layout: int = 0

    def __call__(self, x: torch.Size, layout: str) -> torch.Size:
        return torch.Size((*x, len(layout)))


class CallableListIntAttr:
    """The silent half of the same collision: the class annotation would convert ``crop`` to a string."""
    crop: str = ""

    def __call__(self, x: torch.Size, crop: list[int]) -> torch.Size:
        return torch.Size([x[0], crop[0], crop[1]])
