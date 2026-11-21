# pypto_pro.language.simt.norm4d

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

Computes the Euclidean norm of four components in scaling mode to avoid unnecessary overflow or underflow caused by direct computation of the sum of squares.

$$result = \sqrt{x^2+y^2+z^2+w^2}$$

## Prototype

```python
pypto_pro.language.simt.norm4d(
    x: Scalar,
    y: Scalar,
    z: Scalar,
    w: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| x | Input| Component, which is of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| y | Input| Component, which is of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| z | Input| Component, which is of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| w | Input| Component, which is of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the Euclidean norm of four components. The data type is DT_FP32. If all the four inputs are zero, positive zero is returned. If any input is infinite and even an input is NaN, positive infinity is returned. If no input is infinite and any input is NaN, NaN is returned.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def norm4d_example(
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    z: pl.Tensor[[1, 64], pl.DT_FP32],
    w: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.norm4d(x[0, tid], y[0, tid], z[0, tid], w[0, tid])


@pl.jit()
def norm4d_kernel(
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    z: pl.Tensor[[1, 64], pl.DT_FP32],
    w: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        norm4d_example[64](x, y, z, w, output)
```
