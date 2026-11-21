# pypto_pro.language.simt.sinpi

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

Computes the sine value of the product of $\pi$ and value.

$$result = \sin(\pi\times value)$$

## Prototype

```python
pypto_pro.language.simt.sinpi(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a floating-point scalar expression at runtime. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the sine value. The data type is DT_FP32. The special values are listed below.

| value| Return Value|
|---|---|
| Integer| Zero with the same sign as value|
| +0.5 | +1 |
| +1.5 | -1 |
| ±Inf | NaN |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def sinpi_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.sinpi(value[0, tid])


@pl.jit()
def sinpi_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        sinpi_example[64](value, output)
```
