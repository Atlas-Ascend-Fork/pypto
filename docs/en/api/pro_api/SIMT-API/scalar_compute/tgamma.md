# pypto_pro.language.simt.tgamma

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

Computes the gamma function. For positive real numbers, the definition is as below, and non-integer negative numbers are overwritten by extension.

$$result = \Gamma(value)=\int_0^{+\infty}t^{value-1}e^{-t}\,dt$$

## Prototype

```python
pypto_pro.language.simt.tgamma(
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

Returns the gamma function value. The data type is DT_FP32. For a positive zero, it returns positive infinity, and for a negative zero, it returns negative infinity. For negative integers and negative infinity, it returns NaN. For positive infinity, it returns positive infinity. For NaN, it returns NaN.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def tgamma_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.tgamma(value[0, tid])


@pl.jit()
def tgamma_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        tgamma_example[64](value, output)
```
