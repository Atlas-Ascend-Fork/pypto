# pypto_pro.language.simt.atomic_sub

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

Atomically subtracts the source operand (value) from the destination operand (target).

## Prototype

```python
pypto_pro.language.simt.atomic_sub(
    target: Union[int, float],
    value: Union[int, float],
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| target | Input| Destination operand. The single-element subscript access expression of the tile or tensor must be directly passed, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB tile: Must reside in the UB and use the ND format. DT_INT32, DT_UINT32, and DT_FP32 are supported.<br>- GM tensor: Must use the ND format. DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64 are supported.|
| value | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. It indicates the subtrahend. The data type must be the same as that of target. Numeric literals are processed based on the data type of target. The integer destination operand does not accept floating-point literals.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

The value of target before the update is returned. The return value type is the same as that of target.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def consume_quota(
    remaining: pl.Tensor[[1, 1], pl.DT_UINT32],
    requests: pl.Tensor[[1, 256], pl.DT_UINT32],
    old_remaining: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    old_remaining[0, tid] = pl.simt.atomic_sub(remaining[0, 0], requests[0, tid])


@pl.jit()
def atomic_sub_kernel(
    remaining: pl.Tensor[[1, 1], pl.DT_UINT32],
    requests: pl.Tensor[[1, 256], pl.DT_UINT32],
    old_remaining: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        consume_quota[256](remaining, requests, old_remaining)
```
