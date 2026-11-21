# pypto_pro.language.simt.scalbn

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

Scales a floating-point number by an integer power of 2. It does not compute $2^{exponent}$ separately. The computation formula is as follows:

$$result = value \times 2^{exponent}$$

The computation semantics is the same as that of [pypto_pro.language.simt.ldexp](ldexp.md). The exponent of this API supports DT_INT32 and DT_INT64. The latter supports only DT_INT32.

## Prototype

```python
pypto_pro.language.simt.scalbn(
    value: Scalar,
    exponent: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Value to be scaled, which is of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| exponent | Input| Binary exponent of the scalar type. DT_INT32 and DT_INT64 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the scaling result. The data type is DT_FP32. If value is positive or negative zero, positive or negative infinity, or NaN, value is returned. If exponent is 0, value is returned. If the finite result overflows, the corresponding infinity is returned. If the finite result underflows to zero, zero with the corresponding sign is returned. For subnormal results that can be represented, non-zero values are returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def scalbn_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    exponent: pl.Tensor[[1, 64], pl.DT_INT64],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.scalbn(value[0, tid], exponent[0, tid])


@pl.jit()
def scalbn_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    exponent: pl.Tensor[[1, 64], pl.DT_INT64],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        scalbn_example[64](value, exponent, output)
```
