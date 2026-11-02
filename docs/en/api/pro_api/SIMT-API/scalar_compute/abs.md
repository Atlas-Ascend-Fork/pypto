# pypto_pro.language.simt.abs

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:03:55.355Z pushedAt=2026-10-28T11:57:29.400Z nl=2 -->

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

Computes the absolute value of the source operand. The formula is as follows:

$$result = |value|$$

## Prototype

```python
pypto_pro.language.simt.abs(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type, supporting DT_FP16, DT_BF16, DT_FP32, and DT_INT64. Tensor or Tile elements must be accessed by subscript before being passed. |

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the absolute value of the source operand, with the same data type as the input.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def abs_fp32(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.abs(source[0, tid])


@pl.jit()
def simt_abs_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    with pl.section_vector():
        abs_fp32[256](source, output)
```
