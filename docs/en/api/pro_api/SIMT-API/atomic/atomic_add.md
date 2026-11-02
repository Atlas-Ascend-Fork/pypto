# pypto_pro.language.simt.atomic_add

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:55:57.782Z pushedAt=2026-10-28T11:57:29.340Z nl=2 -->

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

Atomically adds the source operand value to the destination operand target.

## Prototype

```python
pypto_pro.language.simt.atomic_add(
    target: Scalar,
    value: Scalar,
) -> Optional[Scalar]
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | Destination operand of the Scalar type. It must be passed directly as a single-element subscript access expression of a Tile or Tensor, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB Tile: Must reside in UB, use ND, and support DT_INT32, DT_UINT32, DT_FP16, DT_BF16, and DT_FP32.<br>- GM Tensor: Must be ND and support DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_FP32, DT_INT64, and DT_UINT64. |
| value | Input | Source operand of the Scalar type, representing the value to be added. Its data type must be the same as that of target. Numeric literals are processed according to the data type of target, and integer destination operands do not accept floating-point literals. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

- When the data type of target is DT_FP16 or DT_BF16, no value is returned, and the API can only be called as a standalone statement;
- When the data type of target is any other supported data type, the value of target before the update is returned, and the return value type is the same as that of target.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def atomic_add_histogram_gm(
    histogram: pl.Tensor[[1, 16], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_add(histogram[0, tid % 16], 1)


@pl.jit()
def simt_atomic_add_histogram_gm(
    histogram: pl.Tensor[[1, 16], pl.DT_INT32],
):
    with pl.section_vector():
        atomic_add_histogram_gm[256](histogram)
```
