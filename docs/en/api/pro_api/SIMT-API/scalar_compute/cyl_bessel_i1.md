# pypto_pro.language.simt.cyl_bessel_i1

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

Computes the first-order modified Bessel function of the first kind.

$$result = I_1(value)$$

This function is an odd function, which means $I_1(-value)=-I_1(value)$.

## Prototype

```python
pypto_pro.language.simt.cyl_bessel_i1(
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

Returns the first-order modified Bessel function value of the first kind. The data type is DT_FP32.

| value| Return Value|
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | +Inf |
| -Inf | -Inf |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def cyl_bessel_i1_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.cyl_bessel_i1(value[0, tid])


@pl.jit()
def cyl_bessel_i1_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        cyl_bessel_i1_example[64](value, output)
```
