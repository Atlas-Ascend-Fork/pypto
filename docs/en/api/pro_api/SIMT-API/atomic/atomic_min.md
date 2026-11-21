# pypto_pro.language.simt.atomic_min

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

Atomically compares the old value of the destination operand (target) with the source operand (value) and writes the smaller value back to the destination operand. For the destination operand of the DT_FP16 or DT_BF16 data type, only the destination operand is updated, and the old value is not returned.

## Prototype

```python
pypto_pro.language.simt.atomic_min(
    target: Union[int, float],
    value: Union[int, float],
) -> Optional[Scalar]
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| target | Input| Destination operand. The single-element subscript access expression of the tile or tensor must be directly passed, for example, ub_tile[0, 0] or gm_tensor[0, 0].<br>- UB tile: Must reside in the UB and use the ND format. DT_INT32, DT_UINT32, DT_FP16, DT_BF16, and DT_FP32 are supported.<br>- GM tensor: Must use the ND format. DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_FP32, DT_INT64, and DT_UINT64 are supported. |
| value | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. It indicates the value to be compared. The data type must be the same as that of target. Numeric literals are processed based on the data type of target. The integer destination operand does not accept floating-point literals.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

- If the data type of target is DT_FP16 or DT_BF16, there is no return value. This function can only be called as an independent statement.
- If the data type of target is any other supported data type, the value of target before the update is returned. The return value type is the same as that of target.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def reduce_min(
    latency: pl.Tensor[[1, 256], pl.DT_UINT32],
    min_latency: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_min(min_latency[0, 0], latency[0, tid])


@pl.jit()
def atomic_min_kernel(
    latency: pl.Tensor[[1, 256], pl.DT_UINT32],
    min_latency: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    with pl.section_vector():
        reduce_min[256](latency, min_latency)
```
