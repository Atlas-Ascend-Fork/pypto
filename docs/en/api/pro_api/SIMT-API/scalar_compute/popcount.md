# pypto_pro.language.simt.popcount

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

Counts the number of bits whose values are 1 in the binary representation of the source operand. It can be used for bitmap counting or to compute the Hamming distance between two integers in combination with bitwise XOR.

## Prototype

```python
pypto_pro.language.simt.popcount(
    value: int,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be an integer constant or a runtime integer scalar expression. DT_UINT32 and DT_UINT64 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

The return value is a scalar of the DT_INT32 type. The statistical scope varies according to the input type:

| Data Type of value| Statistical Scope| Range of Return Value|
|---|---|---|
| DT_UINT32 | All 32 bits| [0, 32] |
| DT_UINT64 | All 64 bits, including the upper 32 bits| [0, 64] |

## Examples

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def count_bits(
    source: pl.Tensor[[1, THREADS], pl.DT_UINT64],
    output: pl.Tensor[[1, THREADS], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.popcount(source[0, tid])


@pl.jit()
def simt_popcount_kernel(
    source: pl.Tensor[[1, THREADS], pl.DT_UINT64],
    output: pl.Tensor[[1, THREADS], pl.DT_INT32],
):
    with pl.section_vector():
        count_bits[THREADS](source, output)
```
