# pypto_pro.language.simt.cosh

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Computes the hyperbolic cosine of value. The computation formula is as follows:

$$result = \cosh(value) = \frac{e^{value} + e^{-value}}{2}$$

## Prototype

```python
pypto_pro.language.simt.cosh(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the hyperbolic cosine of value. The data type is DT_FP32. The result of a finite input is not less than 1. The special values are listed below.

| value| Return Value|
|---|---|
| ±Inf | +Inf |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def cosh_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.cosh(value[0, tid])


@pl.jit()
def cosh_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        cosh_example[64](value, output)
```
