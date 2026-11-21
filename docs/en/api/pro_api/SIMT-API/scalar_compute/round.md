# pypto_pro.language.simt.round

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

Rounds the source operand to the nearest integer and rounds away from zero if the source operand is exactly halfway between two integers. The computation formula is as follows:

$$result = \begin{cases} \lfloor value + \frac{1}{2} \rfloor & value \geq 0 \\ \lceil value - \frac{1}{2} \rceil & value < 0 \end{cases}$$

## Prototype

```python
pypto_pro.language.simt.round(
    value: float,
    *,
    dtype: DType | None = None,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand of the scalar type. DT_FP16, DT_BF16, and DT_FP32 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| dtype | Input| Optional return data type. The default value is None. It can be set to the same data type as value. When value is DT_FP32, it can also be set to DT_INT64.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- When dtype is set to DT_INT64, value must be DT_FP32.

## Return Value

Returns the rounded integer. If dtype is omitted or set to None, the data type is the same as that of value. If dtype is set to DT_INT64, a DT_INT64 scalar is returned. If the result is out of the range of $[-2^{63},2^{63}-1]$, the result is clamped to the corresponding endpoint.

When dtype is set to DT_INT64, the conversion results of special values are listed below.

| value| Return Value|
|---|---|
| ±0 | 0 |
| +Inf or above the upper bound of DT_INT64| $2^{63}-1$ |
| -Inf or below the lower bound of DT_INT64| $-2^{63}$ |
| NaN | 0 |

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def nearest_away_from_zero(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output_fp32: pl.Tensor[[1, 256], pl.DT_FP32],
    output_int64: pl.Tensor[[1, 256], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    output_fp32[0, tid] = pl.simt.round(source[0, tid])
    output_int64[0, tid] = pl.simt.round(source[0, tid], dtype=pl.DT_INT64)


@pl.jit()
def simt_round_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output_fp32: pl.Tensor[[1, 256], pl.DT_FP32],
    output_int64: pl.Tensor[[1, 256], pl.DT_INT64],
):
    with pl.section_vector():
        nearest_away_from_zero[256](source, output_fp32, output_int64)
```
