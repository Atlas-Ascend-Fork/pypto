# pypto_pro.language.simt.fma

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

Performs fused multiply-add operations. Multiplication and addition are completed in one fused operation, and the intermediate product is not rounded separately. The computation formula is as follows:

$$result = lhs \times rhs + addend$$

## Prototype

```python
pypto_pro.language.simt.fma(
    lhs: float,
    rhs: float,
    addend: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand (multiplier), which can be a floating-point constant or a runtime floating-point scalar expression. DT_FP16, DT_BF16, and DT_FP32 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand (multiplier), which can be a floating-point constant or a runtime floating-point scalar expression. The data type must be the same as that of lhs. The tensor or tile element needs to be accessed by subscript before being passed.|
| addend | Input| Source operand (addend), which can be a floating-point constant or a runtime floating-point scalar expression. The data type must be the same as that of lhs and rhs. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the fused multiply-add result. The data type is the same as that of the inputs.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def linear(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.fma(source[0, tid], scale, bias)


@pl.jit()
def simt_fma_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    with pl.section_vector():
        linear[256](source, output, scale, bias)
```
