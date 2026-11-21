# pypto_pro.language.simt.remainder

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

Computes the IEEE remainder of lhs divided by rhs. Set $n$ as the integer closest to $lhs/rhs$, and choose the even number in case of a tie. The computation formula is as follows:

$$result = lhs-n\times rhs$$

## Prototype

```python
pypto_pro.language.simt.remainder(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Dividend of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Divisor of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the remainder. The data type is DT_FP32. For finite inputs and non-zero divisors, the absolute value of the remainder is no greater than $|rhs|/2$. The remainder can have a different sign from lhs. When the result is zero, the sign of lhs is retained. If the divisor is zero, the dividend is infinite, or either input is NaN, NaN is returned. If the finite dividend is divided by infinity, the dividend is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def remainder_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.remainder(lhs[0, tid], rhs[0, tid])


@pl.jit()
def remainder_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        remainder_example[64](lhs, rhs, output)
```
