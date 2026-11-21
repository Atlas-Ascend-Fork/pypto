# pypto_pro.language.simt.log10

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

Computes the base-10 logarithm of value. The computation formula is as follows:

$$result = \log_{10}(value)$$

## Prototype

```python
pypto_pro.language.simt.log10(
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

Returns the common logarithm of value. The data type is the same as that of the input. The special values are listed below.

| value| Return Value|
|---|---|
| +Inf | +Inf |
| ±0 | -Inf |
| -Inf | NaN |
| Finite negative number| NaN |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def log10_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.log10(value[0, tid])


@pl.jit()
def log10_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        log10_example[64](value, output)
```
