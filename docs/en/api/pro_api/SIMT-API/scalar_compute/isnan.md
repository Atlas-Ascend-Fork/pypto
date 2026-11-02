# pypto_pro.language.simt.isnan

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:08:21.629Z pushedAt=2026-10-28T11:57:29.439Z nl=2 -->

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

Determines whether the source operand is NaN.

## Prototype

```python
pypto_pro.language.simt.isnan(
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

Returns True when the source operand is NaN; otherwise, returns False. The return value is a Scalar of the DT_BOOL type.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def replace_nan(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    fallback: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    value = source[0, tid]
    output[0, tid] = value
    if pl.simt.isnan(value):
        output[0, tid] = fallback


@pl.jit()
def simt_isnan_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    fallback: pl.DT_FP32,
):
    with pl.section_vector():
        replace_nan[256](source, output, fallback)
```
