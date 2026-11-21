# pypto_pro.language.simt.min_nan

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

Compares two low-precision floating-point numbers and returns the smaller value. If either input is NaN, NaN is propagated.

## Prototype

```python
pypto_pro.language.simt.min_nan(
    lhs: float,
    rhs: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. DT_FP16 and DT_BF16 are supported. The data type must be the same as that of other source operands. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. DT_FP16 and DT_BF16 are supported. The data type must be the same as that of other source operands. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- The two inputs must be of the same type, either DT_FP16 or DT_BF16.

## Return Value

Returns the smaller value. The data type is the same as that of the input. If either input is NaN, NaN is returned. If the two inputs are positive zero and negative zero respectively, positive zero is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def min_nan_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.min_nan(lhs[0, tid], rhs[0, tid])


@pl.jit()
def min_nan_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    with pl.section_vector():
        min_nan_example[64](lhs, rhs, output)
```
