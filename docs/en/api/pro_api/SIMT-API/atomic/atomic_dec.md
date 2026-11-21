# pypto_pro.language.simt.atomic_dec

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

Atomically decrements the destination operand (target). If the old value of the destination operand is 0 or greater than the source operand (limit), limit is written to the destination operand. Otherwise, the old value is decremented by 1.

## Prototype

```python
pypto_pro.language.simt.atomic_dec(
    target: Union[int, float],
    limit: Union[int, float],
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| target | Input| Destination operand. The single-element subscript access expression of the tile or tensor must be directly passed, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB tile: Must reside in the UB and use the ND format. DT_UINT32 is supported.<br>- GM tensor: Must be in ND format. DT_UINT32 and DT_UINT64 are supported.|
| limit | Input| Source operand, which can be an integer constant or a runtime integer scalar expression. It indicates the upper bound of the count. The data type must be the same as that of target. Integer literals are processed based on the data type of target.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the value of target before decrement. The return value type is the same as that of target.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def allocate_reverse_ring_slot(
    ticket: pl.Tensor[[1, 1], pl.DT_UINT32],
    slots: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    slots[0, tid] = pl.simt.atomic_dec(ticket[0, 0], 255)


@pl.jit()
def atomic_dec_kernel(
    ticket: pl.Tensor[[1, 1], pl.DT_UINT32],
    slots: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        allocate_reverse_ring_slot[256](ticket, slots)
```
