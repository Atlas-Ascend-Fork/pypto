# pypto_pro.language.simt.tanh

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

Computes the hyperbolic tangent of the source operand. The computation formula is as follows:

$$result = \tanh(value) = \frac{e^{value} - e^{-value}}{e^{value} + e^{-value}}$$

## Prototype

```python
pypto_pro.language.simt.tanh(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a floating-point scalar expression at runtime. The supported data types are DT_FP16, DT_BF16, and DT_FP32. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the hyperbolic tangent of the source operand. The data type is the same as that of the input. The special values are listed below.

| value| Return Value|
|---|---|
| +0 | +0 |
| -0 | +0 |
| +Inf | +1 |
| -Inf | -1 |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def gate(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.tanh(source[0, tid] * scale)


@pl.jit()
def simt_tanh_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
):
    with pl.section_vector():
        gate[256](source, output, scale)
```
