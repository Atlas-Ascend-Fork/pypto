# pypto_pro.language.simt.fma_relu

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

Performs fused multiply-add on three operands, and then clamps the negative result to zero. Intermediate rounding is not performed between multiplication and addition. The computation formula is as follows:

$$result = \max(lhs \times rhs + addend, 0)$$

## Prototype

```python
pypto_pro.language.simt.fma_relu(
    lhs: Scalar,
    rhs: Scalar,
    addend: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand of the scalar type. DT_FP16 and DT_BF16 are supported. The data type must be the same as that of other source operands. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand of the scalar type. DT_FP16 and DT_BF16 are supported. The data type must be the same as that of other source operands. The tensor or tile element needs to be accessed by subscript before being passed.|
| addend | Input| Source operand of the scalar type. DT_FP16 and DT_BF16 are supported. The data type must be the same as that of other source operands. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- The three inputs must be of the same data type, either DT_FP16 or DT_BF16.

## Return Value

Returns the result of the fused multiply-add operation after ReLU processing. The data type is the same as that of the inputs. The result is rounded to the nearest even number. Zero is returned for the negative result, NaN is returned for NaN, and positive infinity is returned for the positive overflow result.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def fma_relu_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    addend: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.fma_relu(lhs[0, tid], rhs[0, tid], addend[0, tid])


@pl.jit()
def fma_relu_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    addend: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    with pl.section_vector():
        fma_relu_example[64](lhs, rhs, addend, output)
```
