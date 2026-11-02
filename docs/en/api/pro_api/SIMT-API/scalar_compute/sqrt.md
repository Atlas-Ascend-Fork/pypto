# pypto_pro.language.simt.sqrt

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:13:35.236Z pushedAt=2026-10-28T11:57:29.484Z nl=2 -->

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

Computes the principal square root of the source operand. The formula is as follows:

$$result = \sqrt{value}$$

## Prototype

```python
pypto_pro.language.simt.sqrt(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type, supporting DT_FP16, DT_BF16, and DT_FP32. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

Can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the principal square root of the source operand, with the same data type as the input. Special values are as follows:

| value | Return Value |
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | +Inf |
| -Inf, NaN, or a finite negative number | NaN |

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def vector_length(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    y: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    squared = x[0, tid] * x[0, tid] + y[0, tid] * y[0, tid]
    output[0, tid] = pl.simt.sqrt(squared)


@pl.jit()
def simt_sqrt_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    y: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    with pl.section_vector():
        vector_length[256](x, y, output)
```
