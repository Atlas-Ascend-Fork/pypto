# pypto_pro.language.simt.nextafter

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

Returns the next representable DT_FP32 value from value in the direction specified by direction. If the values are equal, value is returned.

## Prototype

```python
pypto_pro.language.simt.nextafter(
    value: float,
    direction: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Start value, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| direction | Input| Target direction, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the adjacent representable floating-point value. The data type is DT_FP32. The special values are listed below.

| value and direction| Return Value|
|---|---|
| Either input is NaN.| NaN |
| The values are equal (including zeros with different signs).| value |
| value is ±0, and direction is a positive number.| Smallest positive subnormal number|
| value is ±0, and direction is a negative number.| Smallest negative subnormal number|
| value is +Inf, and direction is a finite number.| Largest finite positive number|
| value is –Inf, and direction is a finite number.| Smallest finite negative number|

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def nextafter_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    direction: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.nextafter(value[0, tid], direction[0, tid])


@pl.jit()
def nextafter_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    direction: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        nextafter_example[64](value, direction, output)
```
