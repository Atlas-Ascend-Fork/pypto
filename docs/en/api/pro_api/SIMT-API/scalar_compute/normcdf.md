# pypto_pro.language.simt.normcdf

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

Computes the cumulative distribution function of the standard normal distribution.

$$result = \Phi(value)=\frac{1}{2}\operatorname{erfc}\left(-\frac{value}{\sqrt{2}}\right)$$

## Prototype

```python
pypto_pro.language.simt.normcdf(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the cumulative probability of the standard normal distribution. The data type is DT_FP32. The value range of the non-NaN result is $[0, 1]$. The value 0.5 is returned for positive or negative zero, 0 for negative infinity, 1 for positive infinity, and NaN for NaN.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def normcdf_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.normcdf(value[0, tid])


@pl.jit()
def normcdf_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        normcdf_example[64](value, output)
```
