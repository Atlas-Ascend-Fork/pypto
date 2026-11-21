# pypto_pro.language.simt.ffs

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

Searches for the first bit whose value is 1 from the least significant bit and returns the 1-based position.

## Prototype

```python
pypto_pro.language.simt.ffs(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand of the scalar type. DT_INT32 and DT_INT64 are supported. The search is performed based on the two's complement bit pattern. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the position of the least significant 1 bit. The data type is DT_INT32. The value ranges from 0 to the input bit width. If value is 0, 0 is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def ffs_example(
    value: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.ffs(value[0, tid])


@pl.jit()
def ffs_kernel(
    value: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    with pl.section_vector():
        ffs_example[64](value, output)
```
