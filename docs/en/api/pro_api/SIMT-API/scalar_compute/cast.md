# pypto_pro.language.simt.cast

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

Converts the source operand to the specified data type. During the conversion, you can specify the rounding mode using the mode parameter.

## Prototype

```python
pypto_pro.language.simt.cast(
    value: Union[int, float],
    dtype: DType,
    *,
    mode: RoundMode = pl.RoundMode.CAST_NONE,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64 are supported. The tensor or tile element needs to be accessed by subscript before being passed.|
| dtype | Input| Destination data type, which is of the DType type. The following table lists the supported data type conversions.|
| mode | Input| (Optional) Rounding mode, which is of the [RoundMode](../../SIMD-API/basic_data_structures/RoundMode.md) type. The default value is pypto_pro.language.RoundMode.CAST_NONE. The following table lists the rounding modes supported by each data type conversion.|

The destination data types and rounding modes supported by each source data type are listed below.

| Source Data Type| Destination Data Type| Supported Mode|
|---|---|---|
| DT_FP16 | DT_FP16, DT_FP32| CAST_NONE |
| DT_FP16 | DT_BF16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_FP16 | DT_INT16, DT_UINT16| CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_FP16 | DT_INT8, DT_UINT8| CAST_TRUNC |
| DT_BF16 | DT_BF16, DT_FP32| CAST_NONE |
| DT_BF16 | DT_FP16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_BF16 | DT_INT16, DT_UINT16| CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_BF16 | DT_INT8, DT_UINT8| CAST_TRUNC |
| DT_FP32 | DT_FP32 | CAST_NONE |
| DT_FP32 | DT_FP16 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC, CAST_ODD|
| DT_FP32 | DT_BF16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64| CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_INT8 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_INT16 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_INT16 | DT_FP16, DT_BF16| CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_INT32 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_INT32 | DT_FP16, DT_BF16| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_INT32 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_INT64 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_INT64 | DT_FP16, DT_BF16| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_INT64 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_UINT8 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_UINT16 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_UINT16 | DT_FP16, DT_BF16| CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_UINT32 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_UINT32 | DT_FP16, DT_BF16| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_UINT32 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_UINT64 | DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64| CAST_NONE |
| DT_UINT64 | DT_FP16, DT_BF16| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|
| DT_UINT64 | DT_FP32 | CAST_NONE, CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

- When DT_FP16 or DT_BF16 is converted to DT_INT8, DT_UINT8, DT_INT16, or DT_UINT16, the rounded result is clamped to the value range of the destination integer type.

## Return Value

Returns the converted scalar. The data type is specified by dtype.

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
