# pypto_pro.language.simt.clz

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

Counts the number of consecutive bits whose values are 0 starting from the most significant bit in the fixed-bit-width binary representation of value.

## Prototype

```python
pypto_pro.language.simt.clz(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand of the scalar type. DT_INT32 and DT_INT64 are supported. The statistics are collected based on the two's complement bit pattern. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the number of leading zeros. The data type is DT_INT32. If the input is zero of the DT_INT32 type, 32 is returned. If the input is zero of the DT_INT64 type, 64 is returned. The most significant bit of a negative number is 1, so 0 is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def clz_example(
    value: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.clz(value[0, tid])


@pl.jit()
def clz_kernel(
    value: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    with pl.section_vector():
        clz_example[64](value, output)
```
