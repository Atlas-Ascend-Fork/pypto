# pypto_pro.language.simt.fmod

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

Computes the floating-point remainder of dividing one DT_FP32 scalar by another. For finite inputs and non-zero rhs, their mathematical semantics is as follows:

$$result = lhs - \operatorname{trunc}(lhs / rhs) \times rhs$$

In the formula, trunc indicates that the intermediate value lhs / rhs is rounded towards zero. The sign of the result is the same as that of the dividend lhs, and the absolute value of the result is less than that of the divisor rhs.

The semantics of this API is different from that of the % operator in Python. For example, fmod(-5.5, 2.0) is -1.5, while the result of the Python expression -5.5 % 2.0 is 0.5.

## Prototype

```python
pypto_pro.language.simt.fmod(
    lhs: float,
    rhs: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Dividend, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Divisor, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a scalar of the DT_FP32 type. The behavior of special inputs is listed below.

| Input| Return Result|
|---|---|
| Any operand is NaN.| NaN |
| lhs is positive or negative infinity, or rhs is positive or negative zero.| NaN |
| lhs is a finite number, and rhs is positive or negative infinity.| lhs, with its sign preserved|
| lhs is positive or negative zero, and rhs is non-zero and non-NaN.| Zero with the same sign as lhs|
| The finite input can be exactly divided, and rhs is non-zero.| Zero with the same sign as lhs|

## Examples

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def calculate_remainder(
    lhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.fmod(lhs[0, tid], rhs[0, tid])


@pl.jit()
def simt_fmod_kernel(
    lhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_FP32],
):
    with pl.section_vector():
        calculate_remainder[THREADS](lhs, rhs, output)
```
