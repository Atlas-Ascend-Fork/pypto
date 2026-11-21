# pypto_pro.language.simt.copysign

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

Returns the combination of the absolute value of magnitude and the sign bit of sign.

## Prototype

```python
pypto_pro.language.simt.copysign(
    magnitude: float,
    sign: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| magnitude | Input| Source operand that provides the absolute value, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| sign | Input| Sign selection operand, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a scalar whose absolute value comes from magnitude and whose sign bit comes from sign. The data type is DT_FP32. If sign is zero, infinity, or NaN, only the sign bit is used. The special values are listed below.

| Value of magnitude| Return Value|
|---|---|
| ±0 | Zero with the sign bit taken from sign|
| ±Inf | Infinity with the sign bit taken from sign|
| NaN | NaN, with the sign bit taken from sign|

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def copysign_example(
    magnitude: pl.Tensor[[1, 64], pl.DT_FP32],
    sign: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.copysign(magnitude[0, tid], sign[0, tid])


@pl.jit()
def copysign_kernel(
    magnitude: pl.Tensor[[1, 64], pl.DT_FP32],
    sign: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        copysign_example[64](magnitude, sign, output)
```
