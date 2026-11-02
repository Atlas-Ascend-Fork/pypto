# pypto_pro.language.simt.cast

<!-- md-trans-meta sourceCommit=a24b692ec106f56c00778fead82f8bef99540dfa translatedAt=2026-10-26T15:04:27.828Z pushedAt=2026-10-28T11:57:29.408Z nl=2 -->

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

Converts the source operand to the specified data type. During conversion, the rounding mode can be specified through the mode parameter.

## Prototype

```python
pypto_pro.language.simt.cast(
    value: Scalar,
    dtype: DType,
    *,
    mode: RoundMode = pl.RoundMode.CAST_NONE,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type, supporting DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64. Tensor or Tile elements must be accessed by subscript before being passed in. |
| dtype | Input | Destination data type of the DType type. For supported data type conversions, see the following table. |
| mode | Input | Optional. Rounding mode of the [RoundMode](../../SIMD-API/basic_data_structures/RoundMode.md) type, defaulting to pypto_pro.language.RoundMode.CAST_NONE. For the rounding modes supported by each data type conversion, see the following table. |

The destination data types and rounding modes supported by each source data type are as follows:

| Source data type | Destination data type | Supported mode |
|---|---|---|
| DT_FP16 | DT_FP16, DT_FP32 | CAST_NONE |
| DT_FP16 | DT_BF16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_FP16 | DT_INT16, DT_UINT16 | CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_FP16 | DT_INT8, DT_UINT8 | CAST_TRUNC |
| DT_BF16 | DT_BF16, DT_FP32 | CAST_NONE |
| DT_BF16 | DT_FP16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_BF16 | DT_INT16, DT_UINT16 | CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_BF16 | DT_INT8, DT_UINT8 | CAST_TRUNC |
| DT_FP32 | DT_FP32 | CAST_NONE |
| DT_FP32 | DT_FP16 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC, CAST_ODD |
| DT_FP32 | DT_BF16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_INT8 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_INT16 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_INT16 | DT_FP16, DT_BF16 | CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_INT32 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_INT32 | DT_FP16, DT_BF16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_INT32 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_INT64 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_INT64 | DT_FP16, DT_BF16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_INT64 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_UINT8 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_UINT16 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_UINT16 | DT_FP16, DT_BF16 | CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_UINT32 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_UINT32 | DT_FP16, DT_BF16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_UINT32 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_UINT64 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64 | CAST_NONE |
| DT_UINT64 | DT_FP16, DT_BF16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |
| DT_UINT64 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC |

## Constraints

- This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

- When DT_FP16 or DT_BF16 is converted to DT_INT8, DT_UINT8, DT_INT16, or DT_UINT16, the rounded result is clamped to the value range of the destination integer type.

## Return Value

Returns the converted Scalar, with the data type specified by dtype.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def cast_floor_fp32_to_int32(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.cast(
        source[0, tid],
        pl.DT_INT32,
        mode=pl.RoundMode.CAST_FLOOR,
    )


@pl.jit()
def simt_cast_floor_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_INT32],
):
    with pl.section_vector():
        cast_floor_fp32_to_int32[256](source, output)
```
