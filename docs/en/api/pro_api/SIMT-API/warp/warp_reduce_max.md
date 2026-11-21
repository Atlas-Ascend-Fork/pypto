# pypto_pro.language.simt.warp_reduce_max

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

Finds the maximum value among the values input by all active threads in a warp. All active threads in the warp return the same result.

## Prototype

```python
pypto_pro.language.simt.warp_reduce_max(
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

## Return Value

Returns the maximum value among the values input by all active lanes. The data type is the same as that of value. All active lanes return the same result.

## Examples

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_reduce_max_example(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_max(values[0, tid])


@pl.jit()
def warp_reduce_max_kernel(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        warp_reduce_max_example[WARP_SIZE](values, output)
```
