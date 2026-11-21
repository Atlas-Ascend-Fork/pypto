# pypto_pro.language.simt.warp_reduce_add

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

Sums up the values of all active lanes in the current warp. All active threads in the warp return the same result.

## Prototype

```python
pypto_pro.language.simt.warp_reduce_add(
    value: Union[int, float],
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Scalar value of the current lane involved in reduction. DT_INT32, DT_UINT32, DT_FP16, and DT_FP32 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- Only the active lanes that have executed the current instruction are reduced. When branch divergence occurs, the participating sets of different branches are different.
- If the sum of data overflows, the computing precision of this API is not ensured.
- This API uses the binary (tree) reduction algorithm at the bottom layer. The computation sequence may be different from that of sequential accumulation. For example, the computation sequence of (((a + b) + c) + d) is different from that of ((a + b) + (c + d)). Floating-point addition is represented with finite precision, and rounding may occur at each step. Therefore, different reduction sequences may produce different intermediate and final results.

## Return Value

Returns the sum of values of all active lanes. The data type is the same as that of value. All active lanes return the same result.

## Examples

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_reduce_add_example(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_add(values[0, tid])


@pl.jit()
def warp_reduce_add_kernel(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        warp_reduce_add_example[WARP_SIZE](values, output)
```
