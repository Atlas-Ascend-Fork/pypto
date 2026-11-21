# pypto_pro.language.simt.expm1

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

Computes $e^{value}-1$. Compared with the scenario where exp is called first and then 1 is subtracted, this API performs precision processing when value is close to zero.

$$result = e^{value} - 1$$

## Prototype

```python
pypto_pro.language.simt.expm1(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Exponent, which can be a floating-point constant or a runtime floating-point scalar expression. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns $e^{value}-1$. The data type is DT_FP32. The special values are listed below.

| value| Return Value|
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | +Inf |
| -Inf | -1 |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def expm1_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.expm1(value[0, tid])


@pl.jit()
def expm1_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        expm1_example[64](value, output)
```
