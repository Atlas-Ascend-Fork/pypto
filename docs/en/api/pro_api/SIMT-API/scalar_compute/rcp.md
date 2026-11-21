# pypto_pro.language.simt.rcp

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

Computes the reciprocal of value. The computation formula is as follows:

$$result = \frac{1}{value}$$

## Prototype

```python
pypto_pro.language.simt.rcp(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. DT_FP16 and DT_BF16 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the reciprocal of value. The data type is the same as that of the input. The special values are listed below.

| value| Return Value|
|---|---|
| +0 | +Inf |
| -0 | -Inf |
| +Inf | +0 |
| -Inf | -0 |
| NaN | NaN |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def rcp_example(
    value: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.rcp(value[0, tid])


@pl.jit()
def rcp_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    with pl.section_vector():
        rcp_example[64](value, output)
```
