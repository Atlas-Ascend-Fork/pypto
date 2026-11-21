# pypto_pro.language.simt.sad

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

Computes the absolute difference between two integers and adds the result to addend. The computation formula is as follows:

$$result = |lhs-rhs|+addend$$

## Prototype

```python
pypto_pro.language.simt.sad(
    lhs: Scalar,
    rhs: Scalar,
    addend: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data type must be the same as that of the other source operand. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data type must be the same as that of the other source operand. The tensor or tile element needs to be accessed by subscript before being passed.|
| addend | Input| Accumulated value of the scalar type. Only DT_UINT32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- lhs and rhs must be of the same type, either DT_INT32 or DT_UINT32. Mixed input is not supported.

## Return Value

Returns the sum of the absolute difference and addend. The data type is DT_UINT32. If the result exceeds the DT_UINT32 range, the result is truncated based on the lower 32 bits.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def sad_example(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    addend: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.sad(lhs[0, tid], rhs[0, tid], addend[0, tid])


@pl.jit()
def sad_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    addend: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    with pl.section_vector():
        sad_example[64](lhs, rhs, addend, output)
```
