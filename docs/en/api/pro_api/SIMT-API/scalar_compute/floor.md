# pypto_pro.language.simt.floor

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:06:34.538Z pushedAt=2026-10-28T11:57:29.430Z nl=2 -->

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

Rounds the source operand toward negative infinity to an integer value. The computation formula is as follows:

$$result = \lfloor value \rfloor$$

## Prototype

```python
pypto_pro.language.simt.floor(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type, supporting DT_FP16, DT_BF16, and DT_FP32. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the largest integer value not greater than the source operand, with the same data type as the input.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def bucket_start(
    position: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.floor(position[0, tid])


@pl.jit()
def simt_floor_kernel(
    position: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    with pl.section_vector():
        bucket_start[256](position, output)
```
