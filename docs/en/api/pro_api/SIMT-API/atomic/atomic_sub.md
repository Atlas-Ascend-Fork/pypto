# pypto_pro.language.simt.atomic_sub

<!-- md-trans-meta sourceCommit=3fcec875f4aab71285ec1f173f266c18da2c066d translatedAt=2026-10-26T14:59:42.472Z pushedAt=2026-10-28T11:57:29.372Z nl=2 -->

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

Atomically subtracts the source operand value from the destination operand target.

## Prototype

```python
pypto_pro.language.simt.atomic_sub(
    target: Scalar,
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | Destination operand of Scalar type. You must pass a single-element subscript access expression of a Tile or Tensor directly, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB Tile: Must reside in UB, use ND, and support DT_INT32, DT_UINT32, and DT_FP32.<br>- GM Tensor: Must be ND and support DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64. |
| value | Input | Source operand of Scalar type, representing the subtrahend. Its data type must be the same as that of target. Numeric literals are processed according to the data type of target, and integer destination operands do not accept floating-point literals. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the value of target before the update. The return value type is the same as that of target.

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
