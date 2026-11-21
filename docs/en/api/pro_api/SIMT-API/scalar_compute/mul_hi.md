# pypto_pro.language.simt.mul_hi

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

Computes the high half of the full product of two integers of the same type. For an input with bit width w, the 2w-bit product is obtained based on the signed or unsigned semantics of the input, and then the high w bits are obtained.

This API is not equivalent to performing multiplication of the input bit width and then shifting the input bit width rightwards. The latter will lose the upper bits during multiplication. This API can be used in scenarios such as multi-precision integer computation.

## Prototype

```python
pypto_pro.language.simt.mul_hi(
    lhs: int,
    rhs: int,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Left operand, which can be an integer constant or a runtime integer scalar expression. DT_INT32, DT_UINT32, DT_INT64, and DT_UINT64 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Right operand, which can be an integer constant or a runtime integer scalar expression. The data type must be the same as that of lhs. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the high half of the full product. The result is a scalar, and the data type is the same as that of the input.

| lhs Data Type| rhs Data Type| Full Product Bit Width| Returned Part| Type of Returned Data|
|---|---|---|---|---|
| DT_INT32 | DT_INT32 | 64-bit| Upper 32 bits| DT_INT32 |
| DT_UINT32 | DT_UINT32 | 64-bit| Upper 32 bits| DT_UINT32 |
| DT_INT64 | DT_INT64 | 128-bit| Upper 64 bits| DT_INT64 |
| DT_UINT64 | DT_UINT64 | 128-bit| Upper 64 bits| DT_UINT64 |

The result of a signed input is interpreted as a signed integer, which is equivalent to performing an arithmetic right shift of w bits on the full signed product.

## Examples

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def multiply_high(
    lhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    output: pl.Tensor[[1, THREADS], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    lhs_value = pl.simt.cast(lhs[0, tid], pl.DT_UINT32)
    rhs_value = pl.simt.cast(rhs[0, tid], pl.DT_UINT32)
    output[0, tid] = pl.simt.mul_hi(lhs_value, rhs_value)


@pl.jit()
def simt_mul_hi_kernel(
    lhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    output: pl.Tensor[[1, THREADS], pl.DT_UINT32],
):
    with pl.section_vector():
        multiply_high[THREADS](lhs, rhs, output)
```
