# pypto_pro.language.simt.signbit

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

Extracts the sign bit of value. This operation directly checks the DT_FP32 bit mode, which can distinguish positive zero from negative zero.

## Prototype

```python
pypto_pro.language.simt.signbit(
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

Returns the sign bit. The data type is DT_INT32. The special values are listed below.

| value| Return Value|
|---|---|
| +0 | 0 |
| +Inf | 0 |
| NaN with the sign bit of 0| 0 |
| -0 | 1 |
| -Inf | 1 |
| NaN with the sign bit of 1| 1 |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def signbit_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.signbit(value[0, tid])


@pl.jit()
def signbit_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    with pl.section_vector():
        signbit_example[64](value, output)
```
