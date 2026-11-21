# pypto_pro.language.simt.log1p

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

Computes the natural logarithm of the source operand plus 1. The computation formula is as follows:

$$result = \ln(1 + value)$$

## Prototype

```python
pypto_pro.language.simt.log1p(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the natural logarithm of the source operand plus 1. The data type is DT_FP32. The special values are listed below.

| value| Return Value|
|---|---|
| +Inf | +Inf |
| -Inf | NaN |
| NaN | NaN |
| -1 | -Inf |
| ±0 | +0 |
| Finite value less than -1| NaN |

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
