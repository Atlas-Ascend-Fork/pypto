# pypto_pro.language.simt.sin

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

Computes the sine value of the source operand. The computation formula is as follows:

$$result = \sin(value)$$

## Prototype

```python
pypto_pro.language.simt.sin(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a floating-point scalar expression at runtime. It indicates the angle to be computed, in radians. The supported data types are DT_FP16, DT_BF16, and DT_FP32. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the sine value of the source operand. The data type is the same as that of the input.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def sine_wave(
    phase: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    offset: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.sin(phase[0, tid] + offset)


@pl.jit()
def simt_sin_kernel(
    phase: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    offset: pl.DT_FP32,
):
    with pl.section_vector():
        sine_wave[256](phase, output, offset)
```
