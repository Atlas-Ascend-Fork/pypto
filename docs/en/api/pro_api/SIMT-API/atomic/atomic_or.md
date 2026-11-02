# pypto_pro.language.simt.atomic_or

<!-- md-trans-meta sourceCommit=3fcec875f4aab71285ec1f173f266c18da2c066d translatedAt=2026-10-26T14:58:58.624Z pushedAt=2026-10-28T11:57:29.361Z nl=2 -->

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

Atomically performs a bitwise OR on the destination operand target and the source operand value, and writes the result back to the destination operand.

## Prototype

```python
pypto_pro.language.simt.atomic_or(
    target: Scalar,
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | Destination operand of Scalar type. You must directly pass a single-element subscript access expression of a Tile or Tensor, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB Tile: Must reside in UB, use ND, and support DT_INT32 and DT_UINT32.<br>- GM Tensor: Must be ND, and support DT_INT32, DT_UINT32, DT_INT64, and DT_UINT64. |
| value | Input | Source operand of Scalar type, representing the bitwise OR mask. The data type must be the same as that of target; integer literals are processed according to the data type of target. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the target value before the update, with the same type as target.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def merge_observed_flags(
    observed_flags: pl.Tensor[[1, 256], pl.DT_UINT32],
    flags: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_or(flags[0, 0], observed_flags[0, tid])


@pl.jit()
def atomic_or_kernel(
    observed_flags: pl.Tensor[[1, 256], pl.DT_UINT32],
    flags: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    with pl.section_vector():
        merge_observed_flags[256](observed_flags, flags)
```
