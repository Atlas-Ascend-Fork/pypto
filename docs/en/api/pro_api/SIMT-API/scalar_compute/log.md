# pypto_pro.language.simt.log

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

Computes the natural logarithm of the source operand. The computation formula is as follows:

$$result = \ln(value)$$

## Prototype

```python
pypto_pro.language.simt.log(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. DT_FP16, DT_BF16, and DT_FP32 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the natural logarithm of the source operand. The data type is the same as that of the input. The special values are listed below.

| value| Return Value|
|---|---|
| +Inf | +Inf |
| -Inf, NaN, or a finite negative number| NaN |
| ±0 | -Inf |

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def safe_log(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    epsilon: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.log(pl.simt.max(source[0, tid], epsilon))


@pl.jit()
def simt_log_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    epsilon: pl.DT_FP32,
):
    with pl.section_vector():
        safe_log[256](source, output, epsilon)
```
