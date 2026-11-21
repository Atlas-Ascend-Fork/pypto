# pypto_pro.language.simt.atomic_add

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

Atomically adds the source operand (value) to the destination operand (target).

## Prototype

```python
pypto_pro.language.simt.atomic_add(
    target: Union[int, float],
    value: Union[int, float],
) -> Optional[Scalar]
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| target | Input| Destination operand. The single-element subscript access expression of the tile or tensor must be directly passed, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB tile: Must reside in the UB and use the ND format. DT_INT32, DT_UINT32, DT_FP16, DT_BF16, and DT_FP32 are supported.<br>- GM tensor: Must use the ND format. DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_FP32, DT_INT64, and DT_UINT64 are supported. |
| value | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. It indicates the accumulated value. The data type must be the same as that of target. Numeric literals are processed based on the data type of target. The integer destination operand does not accept floating-point literals.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

- If the data type of target is DT_FP16 or DT_BF16, there is no return value. This function can only be called as an independent statement.
- If the data type of target is any other supported data type, the value of target before the update is returned. The return value type is the same as that of target.

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
