# pypto_pro.language.simt.log1p

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:09:19.604Z pushedAt=2026-10-28T11:57:29.446Z nl=2 -->

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

Computes the natural logarithm of the source operand plus 1. The formula is as follows:

$$result = \ln(1 + value)$$

## Prototype

```python
pypto_pro.language.simt.log1p(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type. Only DT_FP32 is supported. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the natural logarithm of the source operand plus 1, with the data type DT_FP32. Special values are as follows:

| value | Return Value |
|---|---|
| +Inf | +Inf |
| -Inf or NaN | NaN |
| -1 | -Inf |
| ±0 | +0 |
| Finite value less than -1 | NaN |

The numerical precision of the current implementation is equivalent to first performing a DT_FP32 addition and then computing the natural logarithm of the addition result.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=64)
def log1p_fp32(
    source: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.log1p(source[0, tid])


@pl.jit()
def simt_log1p_kernel(
    source: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        log1p_fp32[64](source, output)
```
