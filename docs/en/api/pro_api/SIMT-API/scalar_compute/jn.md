# pypto_pro.language.simt.jn

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

Computes the non-negative-integer-order Bessel function of the first kind.

$$result = J_{order}(value)$$

## Prototype

```python
pypto_pro.language.simt.jn(
    order: Scalar,
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| order | Input| Order of the scalar type. Only DT_INT32 is supported. The valid order is a non-negative integer. The tensor or tile element needs to be accessed by subscript before being passed.|
| value | Input| Source operand of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- If order is a negative number, NaN is returned, and compile-time validation on the order value is not performed.

## Return Value

Returns the Bessel function value of the first kind for the specified order. The data type is DT_FP32. When order is 0 or 1, the value is the same as that of [pypto_pro.language.simt.j0](j0.md) or [pypto_pro.language.simt.j1](j1.md), respectively.

| Input| Return Value|
|---|---|
| order < 0, or value is NaN| NaN |
| order = 1, value is ±0 or ±Inf| Zero with the same sign as value|
| order ≥ 2, value is ±0 or ±Inf| +0 |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def jn_example(
    order: pl.Tensor[[1, 64], pl.DT_INT32],
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.jn(order[0, tid], value[0, tid])


@pl.jit()
def jn_kernel(
    order: pl.Tensor[[1, 64], pl.DT_INT32],
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        jn_example[64](order, value, output)
```
