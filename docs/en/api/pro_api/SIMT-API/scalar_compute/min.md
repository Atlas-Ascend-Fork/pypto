# pypto_pro.language.simt.min

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:10:42.049Z pushedAt=2026-10-28T11:57:29.459Z nl=2 -->

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

Compares two source operands and returns the smaller value. The computation formula is as follows:

$$result = \min(lhs, rhs)$$

## Prototype

```python
pypto_pro.language.simt.min(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| lhs | Input | Source operand of the Scalar type, supporting DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64. Tensor or Tile elements must be accessed by subscript before being passed in. |
| rhs | Input | Source operand of the Scalar type. The data type must be the same as that of lhs. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

This API can be called only in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the smaller of the two source operands, with the data type consistent with the inputs. For floating-point inputs, if one operand is NaN, the other operand is returned; if both operands are NaN, NaN is returned.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def clamp_upper(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    upper: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.min(source[0, tid], upper)


@pl.jit()
def simt_min_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    upper: pl.DT_FP32,
):
    with pl.section_vector():
        clamp_upper[256](source, output, upper)
```
