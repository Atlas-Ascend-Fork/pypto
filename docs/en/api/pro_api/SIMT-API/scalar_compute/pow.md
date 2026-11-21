# pypto_pro.language.simt.pow

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

Computes the power of lhs as the base and rhs as the exponent.

$$result = lhs^{rhs}$$

## Prototype

```python
pypto_pro.language.simt.pow(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Base number of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Exponent of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- Both inputs must be of the DT_FP32 type. Mixed inputs of integer or low-precision floating-point types are not supported.
- If a finite negative base is combined with a finite non-integer exponent, NaN is returned, and compile-time validation on the input value range is not performed.

## Return Value

Returns the exponentiation operation result. The data type is DT_FP32. The finite integer power of the negative base is negative when the exponent is an odd number and is positive when the exponent is an even number. If the result overflows, infinity with the corresponding sign is returned. If the result underflows to zero, the corresponding sign is retained.

The following rules are matched in sequence in the table.

| Input| Return Value|
|---|---|
| rhs is ±0, or lhs is 1 (the other input can be NaN).| 1 |
| Other inputs containing NaN| NaN |
| lhs is a finite negative number, and rhs is a finite non-integer.| NaN |
| lhs is ±0, rhs > 0| If rhs is an odd integer, zero with the same sign as lhs is returned. Otherwise, +0 is returned.|
| lhs is ±0, rhs < 0| If rhs is an odd integer, infinity with the same sign as lhs is returned. Otherwise, +Inf is returned.|
| lhs is ±Inf, rhs > 0| If rhs is an odd integer, infinity with the same sign as lhs is returned. Otherwise, +Inf is returned.|
| lhs is ±Inf, rhs < 0| If rhs is an odd integer, zero with the same sign as lhs is returned. Otherwise, +0 is returned.|
| lhs is -1, and rhs is ±Inf.| 1 |
| $\vert lhs\vert > 1$, and rhs is +Inf; or $\vert lhs\vert < 1$, and rhs is -Inf| +Inf |
| $\vert lhs\vert < 1$, and rhs is +Inf; or $\vert lhs\vert > 1$, and rhs is -Inf| +0 |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def pow_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.pow(lhs[0, tid], rhs[0, tid])


@pl.jit()
def pow_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        pow_example[64](lhs, rhs, output)
```
