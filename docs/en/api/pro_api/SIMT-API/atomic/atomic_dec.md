# pypto_pro.language.simt.atomic_dec

<!-- md-trans-meta sourceCommit=3fcec875f4aab71285ec1f173f266c18da2c066d translatedAt=2026-10-26T14:56:58.352Z pushedAt=2026-10-28T11:57:29.347Z nl=2 -->

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

Atomically decrements the destination operand target. If the old value of the destination operand is 0 or greater than the source operand limit, limit is written to the destination operand; otherwise, the old value is decremented by 1.

## Prototype

```python
pypto_pro.language.simt.atomic_dec(
    target: Scalar,
    limit: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | Destination operand of Scalar type. You must directly pass a single-element subscript access expression of a Tile or Tensor, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB Tile: Must reside in UB, use ND, and support DT_UINT32.<br>- GM Tensor: Must be ND and support DT_UINT32 and DT_UINT64. |
| limit | Input | Source operand of Scalar type, representing the upper bound of the count. The data type must be the same as that of target; integer literals are processed according to the data type of target. |

## Constraints

This API can be called only in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the value of target before the decrement. The return value type is the same as that of target.

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
