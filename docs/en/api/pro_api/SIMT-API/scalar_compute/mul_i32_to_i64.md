# pypto_pro.language.simt.mul_i32_to_i64

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

Computes the full 64-bit product of two 32-bit integers of the same type.

$$result = lhs \times rhs$$

## Prototype

```python
pypto_pro.language.simt.mul_i32_to_i64(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data types of the two inputs must be the same. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data types of the two inputs must be the same. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- Mixed input of DT_INT32 and DT_UINT32 is not supported.

## Return Value

Returns the full product. The mappings between the input and output types are listed below.

| Data Type of lhs and rhs| Type of Returned Data|
|---|---|
| DT_INT32 | DT_INT64 |
| DT_UINT32 | DT_UINT64 |

All 64 bits of the result are retained. Unlike [pypto_pro.language.simt.mul_hi](mul_hi.md), which returns only the upper half, this function does not truncate the lower 24 bits of the operand as [pypto_pro.language.simt.mul24](mul24.md) does.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def mul_i32_to_i64_example(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.mul_i32_to_i64(lhs[0, tid], rhs[0, tid])


@pl.jit()
def mul_i32_to_i64_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT64],
):
    with pl.section_vector():
        mul_i32_to_i64_example[64](lhs, rhs, output)
```
