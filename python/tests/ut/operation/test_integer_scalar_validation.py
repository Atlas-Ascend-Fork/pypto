#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import pytest

import pypto
from pypto.error import PyptoError


@pytest.mark.parametrize(
    ("op", "scalar_first"),
    [
        (pypto.bitwise_right_shift, False),
        (pypto.bitwise_right_shift, True),
        (pypto.bitwise_left_shift, False),
        (pypto.bitwise_left_shift, True),
        (pypto.gcd, False),
    ],
)
def test_rejects_float_scalar(op, scalar_first):
    tensor = pypto.tensor([1, 16], pypto.DT_INT16)
    operands = (1.5, tensor) if scalar_first else (tensor, 1.5)

    with pytest.raises(PyptoError, match=f"Scalar operand for {op.__name__} must be an integer"):
        op(*operands)
