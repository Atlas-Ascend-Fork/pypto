# pypto_pro.language.simt.fma

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:07:35.649Z pushedAt=2026-10-28T11:57:29.436Z nl=2 -->

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

Performs a fused multiply-add operation, completing multiplication and addition in a single fused operation without rounding the intermediate product separately. The computation formula is as follows:

$$result = lhs \times rhs + addend$$

## Prototype

```python
pypto_pro.language.simt.fma(
    lhs: Scalar,
    rhs: Scalar,
    addend: Scalar,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| lhs | Input | Source operand (multiplier), of Scalar type, supporting DT_FP16, DT_BF16, and DT_FP32. Tensor or Tile elements must be accessed by subscript before being passed in. |
| rhs | Input | Source operand (multiplier), of Scalar type, whose data type must be the same as that of lhs. Tensor or Tile elements must be accessed by subscript before being passed in. |
| addend | Input | Source operand (addend), of Scalar type, whose data type must be the same as those of lhs and rhs. Tensor or Tile elements must be accessed by subscript before being passed in. |

## Constraints

It can be called only in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns the fused multiply-add result, whose data type is the same as that of the inputs.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def linear(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.fma(source[0, tid], scale, bias)


@pl.jit()
def simt_fma_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    with pl.section_vector():
        linear[256](source, output, scale, bias)
```
