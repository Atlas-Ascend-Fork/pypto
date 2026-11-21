# pypto_pro.language.simt.acos

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

Computes the arc cosine of value, and returns the result, in radians.

$$result = \arccos(value)$$

## Prototype

```python
pypto_pro.language.simt.acos(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The operand must be a finite real number within $[-1, 1]$. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the arc cosine value of value. The data type is DT_FP32, and the value range is $[0, \pi]$. The special values are listed below.

| value| Return Value|
|---|---|
| +1 | +0 |
| -1 | $\pi$ |
| Finite number whose absolute value is greater than 1| NaN |
| ±Inf | NaN |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def acos_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.acos(value[0, tid])


@pl.jit()
def acos_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        acos_example[64](value, output)
```
