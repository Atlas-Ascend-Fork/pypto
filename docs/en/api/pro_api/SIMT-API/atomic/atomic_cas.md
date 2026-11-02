# pypto_pro.language.simt.atomic_cas

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:55:48.644Z pushedAt=2026-10-28T11:57:29.334Z nl=2 -->

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

Atomically compares the old value of the destination operand target with the source operand compare. If they are equal, the source operand value is written to the destination operand; otherwise, the destination operand remains unchanged.

## Prototype

```python
pypto_pro.language.simt.atomic_cas(
    target: Scalar,
    compare: Scalar,
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | Destination operand of Scalar type. It must be passed directly as a single-element subscript access expression of a Tile or Tensor, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB Tile: Must reside in UB, use ND, and support DT_INT32, DT_UINT32, and DT_FP32.<br>- GM Tensor: Must be ND and support DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64. |
| compare | Input | Source operand of Scalar type, representing the expected value. Its data type must be the same as that of target. Numeric literals are processed according to the data type of target, and integer destination operands do not accept floating-point literals. |
| value | Input | Source operand of Scalar type, representing the new value to be written when the comparison is equal. Its data type must be the same as that of target. Numeric literals are processed according to the data type of target, and integer destination operands do not accept floating-point literals. |

## Constraints

It can be called only in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the value of target before the comparison, with the return value type the same as that of target. You can determine whether the exchange succeeded by checking whether the return value equals compare.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=32)
def atomic_cas_winner_gm(
    state: pl.Tensor[[1, 1], pl.DT_INT32],
    old_values: pl.Tensor[[1, 32], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_cas(state[0, 0], 0, 1)


@pl.jit()
def simt_atomic_cas_winner_gm(
    state: pl.Tensor[[1, 1], pl.DT_INT32],
    old_values: pl.Tensor[[1, 32], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_cas_winner_gm[32](state, old_values)
```
