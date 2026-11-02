# pypto_pro.language.simt.max

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:10:24.209Z pushedAt=2026-10-28T11:57:29.456Z nl=2 -->

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

Compares two source operands and returns the larger value. The computation formula is as follows:

$$result = \max(lhs, rhs)$$

## Prototype

```python
pypto_pro.language.simt.max(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| lhs | Input | Source operand of the Scalar type. Supported data types: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64. Tensor or Tile elements must be accessed by subscript before being passed in. |
| rhs | Input | Source operand of the Scalar type. The data type must be the same as that of lhs. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the larger of the two source operands, with the same data type as the inputs. For floating-point inputs, if one operand is NaN, the other operand is returned; if both operands are NaN, NaN is returned.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def clamp_lower(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    lower: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.max(source[0, tid], lower)


@pl.jit()
def simt_max_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    lower: pl.DT_FP32,
):
    with pl.section_vector():
        clamp_lower[256](source, output, lower)
```
