# pypto_pro.language.simt.atan2

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

Determines the quadrant based on the signs of y and x, and computes the polar angle of coordinates $(x, y)$ relative to the positive direction of the x-axis. The result is expressed in radians.

$$result = \operatorname{atan2}(y, x)$$

## Prototype

```python
pypto_pro.language.simt.atan2(
    y: float,
    x: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| y | Input| Vertical coordinate, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| x | Input| Horizontal coordinate, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the polar angle of coordinates $(x, y)$. The data type is DT_FP32, and the value range is $[-\pi,\pi]$. The special values are listed below.

| Values of y and x| Return Value|
|---|---|
| Either input is NaN.| NaN |
| y is ±0, and the sign bit of x is 0 (including +0).| y is returned.|
| y is ±0, and the sign bit of x is 1 (including –0).| The absolute value is $\pi$, and the sign is the same as that of y.|
| y is not 0, and x is ±0.| The absolute value is $\frac{\pi}{2}$, and the sign is the same as that of y.|
| y is ±Inf, and x is +Inf.| The absolute value is $\frac{\pi}{4}$, and the sign is the same as that of y.|
| y is ±Inf and x is –Inf.| The absolute value is $\frac{3\pi}{4}$, and the sign is the same as that of y.|

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def atan2_example(
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.atan2(y[0, tid], x[0, tid])


@pl.jit()
def atan2_kernel(
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        atan2_example[64](y, x, output)
```
