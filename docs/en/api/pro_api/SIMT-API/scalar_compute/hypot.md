# pypto_pro.language.simt.hypot

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

Computes the Euclidean norm of two components in scaling mode to avoid unnecessary overflow or underflow caused by direct computation of the sum of squares.

$$result = \sqrt{lhs^2+rhs^2}$$

## Prototype

```python
pypto_pro.language.simt.hypot(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the Euclidean norm of two components. The data type is DT_FP32. If both inputs are zero, positive zero is returned. If either input is infinite and even the other input is NaN, positive infinity is returned. If neither input is infinite and either input is NaN, NaN is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def hypot_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.hypot(lhs[0, tid], rhs[0, tid])


@pl.jit()
def hypot_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        hypot_example[64](lhs, rhs, output)
```
