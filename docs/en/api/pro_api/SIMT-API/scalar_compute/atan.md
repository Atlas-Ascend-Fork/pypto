# pypto_pro.language.simt.atan

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

Computes the arc tangent of value, and returns the result, in radians. The computation formula is as follows:

$$result = \arctan(value)$$

## Prototype

```python
pypto_pro.language.simt.atan(
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

Returns the arc tangent value of value. The data type is DT_FP32, and the value range is $[-\frac{\pi}{2}, \frac{\pi}{2}]$. The special values are listed below.

| value| Return Value|
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | $+\frac{\pi}{2}$ |
| -Inf | $-\frac{\pi}{2}$ |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def atan_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.atan(value[0, tid])


@pl.jit()
def atan_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        atan_example[64](value, output)
```
