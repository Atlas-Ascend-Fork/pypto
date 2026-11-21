# pypto_pro.language.simt.isfinite

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

Determines whether the source operand is a finite floating-point number, which means that it is neither positive infinity nor negative infinity nor NaN.

## Prototype

```python
pypto_pro.language.simt.isfinite(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. DT_FP16 and DT_FP32 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns True if the input is a finite number, including positive and negative zero, normal number, and subnormal number. Returns False if the input is NaN, positive infinity, or negative infinity. The return value is a scalar of the DT_BOOL type.

## Examples

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def classify(
    source: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_BOOL],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.isfinite(source[0, tid])


@pl.jit()
def simt_isfinite_kernel(
    source: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_BOOL],
):
    with pl.section_vector():
        classify[THREADS](source, output)
```
