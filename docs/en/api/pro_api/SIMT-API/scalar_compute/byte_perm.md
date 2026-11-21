# pypto_pro.language.simt.byte_perm

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

Uses lhs as the lower four bytes and rhs as the upper four bytes to form an 8-byte source data, and then uses the selector to select four bytes in sequence to form the result. The $ith$ byte of the result uses the lower three bits of the $ith$ half-byte of the selector as the index. Indexes 0 to 3 correspond to the four bytes of lhs, and indexes 4 to 7 correspond to the four bytes of rhs.

## Prototype

```python
pypto_pro.language.simt.byte_perm(
    lhs: Scalar,
    rhs: Scalar,
    selector: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Lower four source bytes, of the scalar type. Only DT_UINT32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Upper four source bytes, of the scalar type. Only DT_UINT32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| selector | Input| Byte selector, of the scalar type. Only DT_UINT32 is supported. The lower 16 bits are used as four half bytes. The lower three bits of each half byte are valid indexes. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a 4-byte integer sorted by selector. The data type is DT_UINT32.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def byte_perm_example(
    lhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    rhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    selector: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.byte_perm(lhs[0, tid], rhs[0, tid], selector[0, tid])


@pl.jit()
def byte_perm_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    rhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    selector: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    with pl.section_vector():
        byte_perm_example[64](lhs, rhs, selector, output)
```
