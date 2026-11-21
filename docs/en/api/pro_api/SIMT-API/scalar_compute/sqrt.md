# pypto_pro.language.simt.sqrt

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

Returns the principal square root of the source operand. The data type is the same as that of the input. The special values are listed below.

| value| Return Value|
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | +Inf |
| -Inf, NaN, or finite negative number| NaN |

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
