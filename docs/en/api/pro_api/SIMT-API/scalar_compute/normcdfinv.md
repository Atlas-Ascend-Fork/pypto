# pypto_pro.language.simt.normcdfinv

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

Computes the inverse function of the cumulative distribution function of the standard normal distribution. It mathematically satisfies $\Phi(result)=value$, and returns its floating-point approximation.

## Prototype

```python
pypto_pro.language.simt.normcdfinv(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Cumulative probability, which is of the scalar type. Only DT_FP32 is supported. The domain of real numbers is $[0, 1]$. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- The domain is described in mathematical semantics, and compile-time validation is not performed on the input value range. If the input is out of the range $[0, 1]$ (including positive and negative infinity) or is NaN, NaN is returned.

## Return Value

Returns the standard normal distribution quantile. The data type is DT_FP32. If the input is 0 or 1, negative infinity or positive infinity is returned, respectively. If the input is 0.5, zero is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def normcdfinv_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.normcdfinv(value[0, tid])


@pl.jit()
def normcdfinv_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        normcdfinv_example[64](value, output)
```
