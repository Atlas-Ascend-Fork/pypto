# pypto_pro.language.simt.cos

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:05:07.357Z pushedAt=2026-10-28T11:57:29.422Z nl=2 -->

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

Computes the cosine of the source operand. The formula is as follows:

$$result = \cos(value)$$

## Prototype

```python
pypto_pro.language.simt.cos(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type, representing the angle to be computed in radians. DT_FP16, DT_BF16, and DT_FP32 are supported. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

This API can be called only in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the cosine of the source operand, with the same data type as the input.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def cosine_wave(
    phase: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    offset: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.cos(phase[0, tid] + offset)


@pl.jit()
def simt_cos_kernel(
    phase: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    offset: pl.DT_FP32,
):
    with pl.section_vector():
        cosine_wave[256](phase, output, offset)
```
