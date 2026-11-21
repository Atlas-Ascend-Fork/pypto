# pypto_pro.language.simt.isinf

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

Determines whether the source operand is positive or negative infinity.

## Prototype

```python
pypto_pro.language.simt.isinf(
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

Returns True if the source operand is positive or negative infinity. Otherwise, returns False. The return value is a scalar of the DT_BOOL type.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def exp_or_fallback(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    fallback: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    result = pl.simt.exp(source[0, tid])
    output[0, tid] = result
    if pl.simt.isinf(result):
        output[0, tid] = fallback


@pl.jit()
def simt_isinf_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    fallback: pl.DT_FP32,
):
    with pl.section_vector():
        exp_or_fallback[256](source, output, fallback)
```
