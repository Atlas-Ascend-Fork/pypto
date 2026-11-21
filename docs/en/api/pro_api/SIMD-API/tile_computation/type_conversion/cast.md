# pypto_pro.language.cast

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

Performs element-wise data type conversion on the elements within the valid region of the source Tile, and writes the results to the corresponding positions in the destination Tile. The destination data type is determined by out.dtype:

$$
out_{i,j}=\operatorname{cast}_{mode}(src_{i,j})
$$

When the conversion may lose precision, the rounding mode can be specified via mode. The input values must be within the representable range of the destination data type.

## Prototype

```python
pypto_pro.language.cast(
    out: Tile,
    src: Tile,
    *,
    mode: RoundMode = pl.RoundMode.CAST_ROUND,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, which must be a UB Tile. The destination data type is determined by out.dtype. out.valid_shape specifies the valid region for this conversion. src must contain valid elements corresponding to this region, and the shapes and layouts of the source and destination Tiles must meet the requirements for element-wise access. For details about the supported data type combinations, see [Supported Data Type Conversions](#supported-data-type-conversions). After the call is complete, the valid region of out is overwritten by the conversion result.|
| src | Input| Source operand, which must be a UB Tile. Its valid region must cover the conversion region specified by out.valid_shape, and every element within that region must be accessible through the same logical coordinates in out. The physical shapes and row strides of the source and destination Tiles may differ, but both must provide sufficient storage space for the conversion region.|
| mode | Input| (Optional) [pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md) enumeration type. It must be a compile-time determinable RoundMode enumeration value. The default value is pypto_pro.language.RoundMode.CAST_ROUND. This parameter affects the result only when the conversion path involves rounding. For the semantics of each enumeration value, see RoundMode; for the support of different conversion paths, see [Rounding Modes](#rounding-modes) and [Supported Data Type Conversions](#supported-data-type-conversions).|

## Constraints

### API usage constraints

- **The destination data type is not passed through a parameter**. This API has no dtype or target_type parameter; the destination data type is determined by the pre-created out Tile.
- **The API does not return a new Tile**. The conversion result is written to out, and the API returns None. It cannot be written in the Tensor API style as dst = pypto_pro.language.cast(src, dtype).
- **Same-type conversion is not a general copy API**. Currently, only same-type conversion from DT_FP32 to DT_FP32 is supported, and this path performs a rounding operation rather than a raw-value copy. Other same-type combinations are not supported.

### Rounding Modes

For the general semantics of the enumeration values of RoundMode, see [pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md). This API also has the following rules related to data type conversion paths:

- For type-widening or integer-narrowing paths that do not involve rounding, mode does not participate in result computation; CAST_NONE is recommended to express this intent.
- The DT_FP16/DT_FP32 → DT_HF8 path always uses CAST_ROUND; passing other modes does not change the rounding behavior of this path.
- Except for DT_FP32 → DT_FP16, passing CAST_ODD on other paths does not perform odd rounding; instead, it falls back to the default rule corresponding to CAST_NONE: floating-point-to-integer conversion uses CAST_TRUNC, and other paths use CAST_RINT.

### Supported Data Type Conversions

The supported combinations of source/destination data types are shown in the table below; combinations not listed are not supported. The "Supported Valid Rounding Mode" in the table refer to modes that can actually change the rounding result; CAST_NONE can be used for all listed conversions and is handled according to the rules described in the previous section. The API passes CAST_ROUND by default.

| Source Data Type| Destination Data Type| Supported Valid Rounding Mode| Remarks|
|---|---|---|---|
| DT_FP32 | DT_FP32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| Same-type rounding, not a raw-value copy.|
| DT_FP32 | DT_FP16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC, and CAST_ODD| - |
| DT_FP32 | DT_BF16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| - |
| DT_FP32 | DT_INT16, DT_INT32, and DT_INT64| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| CAST_NONE uses CAST_TRUNC.|
| DT_FP32 | DT_FP8E4M3FN and DT_FP8E5M2| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| - |
| DT_FP32 | DT_HF8 | Fixed to CAST_ROUND| Other modes do not change the rounding behavior.|
| DT_FP16 | DT_FP32 | No rounding involved| Type widening. mode does not affect the result.|
| DT_FP16 | DT_INT8, DT_UINT8, DT_INT16, and DT_INT32| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| CAST_NONE uses CAST_TRUNC.|
| DT_FP16 | DT_HF8 | Fixed to CAST_ROUND| Other modes do not change the rounding behavior.|
| DT_BF16 | DT_FP32 | No rounding involved| Type widening. mode does not affect the result.|
| DT_BF16 | DT_INT32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| CAST_NONE uses CAST_TRUNC.|
| DT_BF16 | DT_FP16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| - |
| DT_BF16 | DT_FP4E2M1 and DT_FP4E1M2| CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| FP4 is a packed type; two elements share one byte.|
| DT_FP4E2M1 and DT_FP4E1M2| DT_BF16 | No rounding involved| Type widening. mode does not affect the result.|
| DT_UINT8 | DT_FP16 and DT_UINT16| No rounding involved| Type widening. mode does not affect the result.|
| DT_INT8 | DT_FP16, DT_INT16, and DT_INT32| No rounding involved| Type widening. mode does not affect the result.|
| DT_INT16 | DT_FP16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| - |
| DT_INT16 | DT_FP32, DT_UINT32, and DT_INT32| No rounding involved| Type widening. mode does not affect the result.|
| DT_INT16 | DT_UINT8 | No rounding involved| Integer narrowing. mode does not affect the result.|
| DT_INT32 | DT_FP32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| - |
| DT_INT32 | DT_INT64 | No rounding involved| Type widening. mode does not affect the result.|
| DT_INT32 | DT_UINT8, DT_INT16, and DT_UINT16| No rounding involved| Integer narrowing. mode does not affect the result.|
| DT_UINT32 | DT_UINT8, DT_UINT16, and DT_INT16| No rounding involved| Integer narrowing. mode does not affect the result.|
| DT_INT64 | DT_FP32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, and CAST_TRUNC| - |
| DT_INT64 | DT_INT32 | No rounding involved| Integer narrowing. mode does not affect the result.|
| DT_FP8E4M3FN, DT_FP8E5M2, and DT_HF8| DT_FP32 | No rounding involved| Type widening. mode does not affect the result.|

### Other Constraints

- Both out and src must be UB Tiles. The conversion range is determined by out.valid_shape. The valid region of src must cover this range. The physical shapes and row strides of the source and destination tiles can be different, but the elements in the conversion region must correspond to each other based on the same logical coordinates. The API defines conversion results only within the valid region of out; content outside this region must not be relied upon.
- For DT_FP32 → DT_FP16, in-place conversion is supported when out and src use exactly the same UB start address. Apart from this scenario, conversion results are not guaranteed when the source and destination storage regions partially or completely overlap. When other data type combinations are used, out and src should be allocated non-overlapping UB storage regions.
- For DT_FP4E2M1 and DT_FP4E1M2, the Tile shape counts logical elements, with two adjacent elements packed into one byte; therefore, the last dimension of the physical shape must be a multiple of 2. For DT_BF16-to-FP4 conversion, the last dimension of valid_shape should also be even, so that every valid element can form a complete packed byte.
- mode must be determined at compile time and cannot be dynamically selected using runtime Scalar or Tensor values.

## Return Value

None

## Examples

### FP16→FP32

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def cast_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    src_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    out_type = pl.TileType(
        shape=[64, 128],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    src_tile = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    out_tile = pl.make_tile_group(type=out_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        src_current = src_tile.current()
        out_current = out_tile.current()
        pl.load(src_current, src, [0, 0])
        pl.cast(out_current, src_current, mode=pl.RoundMode.CAST_NONE)
        pl.store(out, out_current, [0, 0])
```

Result example:

<!-- pypto-doc-output:cast:start -->
```bash
Input data src: [[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
Output data out: [[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
```
<!-- pypto-doc-output:cast:end -->

### Rounding Modes

```python
# FP32 -> FP16: Round to the nearest value, and round to the nearest even value in case of a tie.
pl.cast(dst_fp16, src_fp32, mode=pl.RoundMode.CAST_RINT)

# FP32 -> INT32: Round towards negative infinity.
# For example, 1.6 is rounded to 1, and -1.6 is rounded to -2.
pl.cast(dst_int32, src_fp32, mode=pl.RoundMode.CAST_FLOOR)

# FP16 -> FP32: No rounding required (type widening)
pl.cast(dst_fp32, src_fp16, mode=pl.RoundMode.CAST_NONE)
```
