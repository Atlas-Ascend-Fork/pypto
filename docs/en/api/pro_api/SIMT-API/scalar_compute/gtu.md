# pypto_pro.language.simt.gtu

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

Performs a greater than or unordered comparison. Returns true if either lhs or rhs is NaN or the relationship lhs > rhs holds. Otherwise, returns false.

## Prototype

```python
pypto_pro.language.simt.gtu(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand of the scalar type. DT_FP16 and DT_BF16 are supported. The data type must be the same as that of other source operands. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand of the scalar type. DT_FP16 and DT_BF16 are supported. The data type must be the same as that of other source operands. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- The two inputs must be of the same type, either DT_FP16 or DT_BF16.

## Return Value

Returns the comparison result. The data type is DT_BOOL.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def gtu_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_BOOL],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.gtu(lhs[0, tid], rhs[0, tid])


@pl.jit()
def gtu_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_BOOL],
):
    with pl.section_vector():
        gtu_example[64](lhs, rhs, output)
```
