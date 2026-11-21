# pypto_pro.language.simt.lgamma

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

Computes the natural logarithm of the absolute value of the gamma function.

$$result = \ln\left|\Gamma(value)\right|$$

## Prototype

```python
pypto_pro.language.simt.lgamma(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- The gamma function has poles at zero and negative integers.

## Return Value

Returns the natural logarithm of the absolute value of the gamma function. The data type is DT_FP32. If the input is zero, a negative integer, or positive/negative infinity, positive infinity is returned. If the input is NaN, NaN is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def lgamma_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lgamma(value[0, tid])


@pl.jit()
def lgamma_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        lgamma_example[64](value, output)
```
