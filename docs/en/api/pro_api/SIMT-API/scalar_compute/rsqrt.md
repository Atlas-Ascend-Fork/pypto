# pypto_pro.language.simt.rsqrt

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:12:42.066Z pushedAt=2026-10-28T11:57:29.469Z nl=2 -->

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

Computes the reciprocal of the square root of the source operand. The formula is as follows:

$$result = \frac{1}{\sqrt{value}}$$

## Prototype

```python
pypto_pro.language.simt.rsqrt(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type. Supports DT_FP16, DT_BF16, and DT_FP32. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the reciprocal of the square root of the source operand, with the same data type as the input. Special values are as follows:

| value | Return Value |
|---|---|
| +0 | +Inf |
| -0 | -Inf |
| +Inf | +0 |
| -Inf, NaN, or a finite negative number | NaN |

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def normalize_scale(
    sum_of_squares: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    epsilon: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    safe_value = pl.simt.max(sum_of_squares[0, tid], epsilon)
    output[0, tid] = pl.simt.rsqrt(safe_value)


@pl.jit()
def simt_rsqrt_kernel(
    sum_of_squares: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    epsilon: pl.DT_FP32,
):
    with pl.section_vector():
        normalize_scale[256](sum_of_squares, output, epsilon)
```
