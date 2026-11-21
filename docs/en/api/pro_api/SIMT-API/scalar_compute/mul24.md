# pypto_pro.language.simt.mul24

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

Multiplies the lower 24 bits of two 32-bit inputs and returns the lower 32 bits of the product. For DT_INT32, the lower 24 bits are sign-extended as 24-bit signed integers. For DT_UINT32, the lower 24 bits are interpreted as unsigned integers.

## Prototype

```python
pypto_pro.language.simt.mul24(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data type must be the same as that of the other source operand. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data type must be the same as that of the other source operand. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- Mixed input of DT_INT32 and DT_UINT32 is not supported.

## Return Value

Returns the lower 32 bits of the product of the operands' lower 24 bits. The data type is the same as that of the input.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def mul24_example(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.mul24(lhs[0, tid], rhs[0, tid])


@pl.jit()
def mul24_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    with pl.section_vector():
        mul24_example[64](lhs, rhs, output)
```
