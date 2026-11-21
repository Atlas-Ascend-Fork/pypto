# pypto_pro.language.simt.saturate

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

Clamps a FP32 scalar to $[0, 1]$ and maps NaN to zero.

## Prototype

```python
pypto_pro.language.simt.saturate(
    value: Scalar,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand of the scalar type. Only DT_FP32 is supported. The tensor or tile element needs to be accessed by subscript before being passed.|

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a scalar of the DT_FP32 type. The value range is $[0, 1]$.

| value| Return Value|
|---|---|
| value ≤ 0 (including -Inf)| 0 |
| 0 < value < 1 | value |
| value ≥ 1 (including +Inf)| 1 |
| NaN | 0 |

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def saturate_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.saturate(value[0, tid])


@pl.jit()
def saturate_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        saturate_example[64](value, output)
```
