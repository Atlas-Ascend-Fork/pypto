# pypto_pro.language.simt.rhadd

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

Computes the rounded-up average of two integers while avoiding intermediate sum overflow.

$$result = \left\lceil\frac{lhs+rhs}{2}\right\rceil$$

## Prototype

```python
pypto_pro.language.simt.rhadd(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data type must be the same as that of the other source operand. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand of the scalar type. DT_INT32 and DT_UINT32 are supported. The data type must be the same as that of the other source operand. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- Mixed input of DT_INT32 and DT_UINT32 is not supported.

## Return Value

Returns the rounded-up average value of the two inputs. The data type is the same as that of the inputs.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def rhadd_example(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.rhadd(lhs[0, tid], rhs[0, tid])


@pl.jit()
def rhadd_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    with pl.section_vector():
        rhadd_example[64](lhs, rhs, output)
```
