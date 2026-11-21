# pypto_pro.language.simt.min

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

Compares two source operands and returns the smaller value. The computation formula is as follows:

$$result = \min(lhs, rhs)$$

## Prototype

```python
pypto_pro.language.simt.min(
    lhs: Union[int, float],
    rhs: Union[int, float],
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| rhs | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. The data type must be the same as that of lhs. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the smaller value between the two source operands. The data type is the same as that of the input. For floating-point inputs, if one operand is NaN, the other operand is returned. If both operands are NaN, NaN is returned.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def clamp_upper(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    upper: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.min(source[0, tid], upper)


@pl.jit()
def simt_min_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    upper: pl.DT_FP32,
):
    with pl.section_vector():
        clamp_upper[256](source, output, upper)
```
