# pypto_pro.language.simt.atomic_max

<!-- md-trans-meta sourceCommit=3fcec875f4aab71285ec1f173f266c18da2c066d translatedAt=2026-10-26T14:58:22.141Z pushedAt=2026-10-28T11:57:29.357Z nl=2 -->

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

Atomically compares the old value of the destination operand target with the source operand value, and writes the larger value back to the destination operand. For destination operands of data type DT_FP16 and DT_BF16, only the destination operand is updated, and the old value is not returned.

## Prototype

```python
pypto_pro.language.simt.atomic_max(
    target: Scalar,
    value: Scalar,
) -> Optional[Scalar]
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | Destination operand of Scalar type. You must directly pass a single-element subscript access expression of a Tile or Tensor, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB Tile: Must reside in UB, use ND, and support DT_INT32, DT_UINT32, DT_FP16, DT_BF16, and DT_FP32.<br>- GM Tensor: Must be ND, and support DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_FP32, DT_INT64, and DT_UINT64. |
| value | Input | Source operand of Scalar type, representing the comparison value. The data type must be the same as that of target. Numeric literals are processed according to the data type of target, and integer destination operands do not accept floating-point literals. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

- When the data type of target is DT_FP16 or DT_BF16, no value is returned, and it can only be called as a standalone statement.
- When the data type of target is any other supported data type, the value of target before the update is returned, and the return value type is the same as that of target.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def reduce_max(
    scores: pl.Tensor[[1, 256], pl.DT_INT32],
    max_score: pl.Tensor[[1, 1], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_max(max_score[0, 0], scores[0, tid])


@pl.jit()
def atomic_max_kernel(
    scores: pl.Tensor[[1, 256], pl.DT_INT32],
    max_score: pl.Tensor[[1, 1], pl.DT_INT32],
):
    with pl.section_vector():
        reduce_max[256](scores, max_score)
```
