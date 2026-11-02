# pypto_pro.language.simt.atomic_exch

<!-- md-trans-meta sourceCommit=3fcec875f4aab71285ec1f173f266c18da2c066d translatedAt=2026-10-26T14:57:29.434Z pushedAt=2026-10-28T11:57:29.350Z nl=2 -->

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

Atomically writes the source operand value to the destination operand target.

## Prototype

```python
pypto_pro.language.simt.atomic_exch(
    target: Scalar,
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | Destination operand of Scalar type. It must be passed directly as a single-element subscript access expression of a Tile or Tensor, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB Tile: Must reside in UB, use ND, and support DT_INT32, DT_UINT32, and DT_FP32.<br>- GM Tensor: Must be ND and support DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64. |
| value | Input | Source operand of Scalar type, representing the new value to be written. Its data type must be the same as that of target. Numeric literals are processed according to the data type of target, and integer destination operands do not accept floating-point literals. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the value of target before the write, with the same type as target.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def publish_fault(
    fault_flags: pl.Tensor[[1, 256], pl.DT_UINT32],
    status: pl.Tensor[[1, 1], pl.DT_UINT32],
    old_status: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    if fault_flags[0, tid] != 0:
        old_status[0, tid] = pl.simt.atomic_exch(status[0, 0], 1)


@pl.jit()
def atomic_exch_kernel(
    fault_flags: pl.Tensor[[1, 256], pl.DT_UINT32],
    status: pl.Tensor[[1, 1], pl.DT_UINT32],
    old_status: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        publish_fault[256](fault_flags, status, old_status)
```
