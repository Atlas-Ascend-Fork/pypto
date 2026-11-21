# pypto_pro.language.simt.ceil

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

Rounds the source operand to the nearest integer towards positive infinity. The computation formula is as follows:

$$result = \lceil value \rceil$$

## Prototype

```python
pypto_pro.language.simt.ceil(
    value: float,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be a floating-point constant or a runtime floating-point scalar expression. DT_FP16, DT_BF16, and DT_FP32 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the smallest integer that is not less than the source operand. The data type is the same as that of the input.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def block_count(
    lengths: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    block_size: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.ceil(lengths[0, tid] / block_size)


@pl.jit()
def simt_ceil_kernel(
    lengths: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
    block_size: pl.DT_FP32,
):
    with pl.section_vector():
        block_count[256](lengths, output, block_size)
```
