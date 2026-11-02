# pypto_pro.language.simt.log2

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:09:47.851Z pushedAt=2026-10-28T11:57:29.449Z nl=2 -->

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

Computes the base-2 logarithm of the source operand. The formula is as follows:

$$result = \log_{2}(value)$$

## Prototype

```python
pypto_pro.language.simt.log2(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type, supporting DT_FP16, DT_BF16, and DT_FP32. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

This API can be called only within a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the base-2 logarithm of the source operand, with the same data type as the input. The special values are as follows:

| value | Return Value |
|---|---|
| +Inf | +Inf |
| -Inf, NaN, or a finite negative number | NaN |
| ±0 | -Inf |

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def binary_log(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    epsilon: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.log2(pl.simt.max(source[0, tid], epsilon))


@pl.jit()
def simt_log2_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    epsilon: pl.DT_FP32,
):
    with pl.section_vector():
        binary_log[256](source, output, epsilon)
```
