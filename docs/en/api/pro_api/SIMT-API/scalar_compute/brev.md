# pypto_pro.language.simt.brev

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

Reverses the bit order of value. When the input bit width is $N$, the original $ith$ bit is moved to the $(N-1-i)th$ bit.

## Prototype

```python
pypto_pro.language.simt.brev(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand of the scalar type. DT_UINT32 and DT_UINT64 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns an integer with the bit order reversed. The data type is the same as that of the input.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def brev_example(
    value: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.brev(value[0, tid])


@pl.jit()
def brev_kernel(
    value: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    with pl.section_vector():
        brev_example[64](value, output)
```
