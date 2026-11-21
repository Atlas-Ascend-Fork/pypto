# pypto_pro.language.simt.atomic_cas

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

Atomically compares the old value of the destination operand (target) with the source operand (compare). If they are equal, the source operand (value) is written to the destination operand. Otherwise, the destination operand remains unchanged.

## Prototype

```python
pypto_pro.language.simt.atomic_cas(
    target: Union[int, float],
    compare: Union[int, float],
    value: Union[int, float],
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| target | Input| Destination operand. The single-element subscript access expression of the tile or tensor must be directly passed, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB tile: Must reside in the UB and use the ND format. DT_INT32, DT_UINT32, and DT_FP32 are supported.<br>- GM tensor: Must use the ND format. DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64 are supported.|
| compare | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. It indicates the expected value. The data type must be the same as that of target. Numeric literals are processed based on the data type of target. The integer destination operand does not accept floating-point literals.|
| value | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. It indicates the new value to be written when the values are equal. The data type must be the same as that of target. Numeric literals are processed based on the data type of target. The integer destination operand does not accept floating-point literals.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

The value of target before the comparison is returned. The return value type is the same as that of target. You can determine whether the swap is successful by checking whether the return value is equal to compare.

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
