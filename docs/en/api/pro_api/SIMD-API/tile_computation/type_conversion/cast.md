# pypto_pro.language.cast

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:50:45.424Z pushedAt=2026-10-28T11:57:29.311Z nl=2 -->

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

Performs element-wise data type conversion on the elements within the valid region of the source Tile and writes the results to the corresponding positions of the target Tile. The target data type is determined by out.dtype:

$$
out_{i,j}=\operatorname{cast}_{mode}(src_{i,j})
$$

When the conversion causes precision loss, you can specify a rounding mode through mode. The input values must be within the representable range of the target data type.

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

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Target operand, which must be a UB Tile. The target data type is determined by out.dtype. out.valid_shape specifies the valid region for this conversion; src must contain the valid elements corresponding to this region, and the shapes and layouts of the source and target Tiles must satisfy the element-wise corresponding access requirement. For supported data type combinations, see [Supported Data Type Conversions](#supported-data-type-conversions). After the call completes, the valid region of out is overwritten by the conversion results. |
| src | Input | Source operand, which must be a UB Tile. Its valid region must cover the conversion region specified by out.valid_shape and ensure that every element in this region can be accessed at the same logical coordinates in out. The physical shapes and row strides of the source and target Tiles may differ, but both must provide sufficient storage for the conversion region. |
| mode | Input | Optional. An enumeration of [pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md), which must be a RoundMode enumeration value determinable at compile time. The default value is pypto_pro.language.RoundMode.CAST_ROUND. This parameter affects the result only when the conversion path involves rounding. For the semantics of each enumeration value, see RoundMode. For the support of different conversion paths, see [Rounding Mode](#rounding-mode) and [Supported Data Type Conversions](#supported-data-type-conversions). |

## Constraints

### API Usage Constraints

- **The target data type is not passed as a parameter**: This API has no dtype or target_type parameter. The target data type is determined by the pre-created out Tile.
- **The API does not return a new Tile**: The conversion result is written to out, and the API returns None. You cannot write dst = pypto_pro.language.cast(src, dtype) as you would with a Tensor API.
- **Same-type conversion is not a general-purpose copy API**: Currently, only DT_FP32-to-DT_FP32 same-type conversion is supported. This path performs a rounding operation rather than copying the original value. Other same-type combinations are not supported.

### Rounding Mode

For the general semantics of each RoundMode enumeration value, see [pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md). This API also has the following rules related to data type conversion paths:

- For type extension or integer narrowing paths with no rounding involved, mode does not participate in result computation. Use CAST_NONE to express the intent.
- The DT_FP16/DT_FP32 -> DT_HF8 path always uses CAST_ROUND. Passing other modes does not change the rounding behavior of this path.
- Except for DT_FP32 -> DT_FP16, passing CAST_ODD to other paths does not perform odd rounding. Instead, it falls back to the default rule corresponding to CAST_NONE: floating-point to integer conversion uses CAST_TRUNC, and other paths use CAST_RINT.

### Supported Data Type Conversions

The supported source/destination data type combinations are listed in the following table. Combinations not listed are not supported. In the table, "Supported Valid Rounding Modes" indicates the modes that can actually change the rounding result. CAST_NONE can be used for all listed conversions and is handled according to the rules described in the previous section. The API passes CAST_ROUND by default.

| Source Data Type | Destination Data Type | Supported Valid Rounding Modes | Remarks |
|---|---|---|---|
| DT_FP32 | DT_FP32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | Rounding of the same type, not a copy of the original value |
| DT_FP32 | DT_FP16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC, CAST_ODD | - |
| DT_FP32 | DT_BF16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | - |
| DT_FP32 | DT_INT16, DT_INT32, DT_INT64 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | CAST_NONE uses CAST_TRUNC |
| DT_FP32 | DT_FP8E4M3FN, DT_FP8E5M2 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | - |
| DT_FP32 | DT_HF8 | Fixed CAST_ROUND | Other modes do not change the rounding behavior |
| DT_FP16 | DT_FP32 | No rounding involved | Type extension; mode does not affect the result |
| DT_FP16 | DT_INT8, DT_UINT8, DT_INT16, DT_INT32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | CAST_NONE uses CAST_TRUNC |
| DT_FP16 | DT_HF8 | Fixed CAST_ROUND | Other modes do not change the rounding behavior |
| DT_BF16 | DT_FP32 | No rounding involved | Type extension; mode does not affect the result |
| DT_BF16 | DT_INT32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | CAST_NONE uses CAST_TRUNC |
| DT_BF16 | DT_FP16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | - |
| DT_BF16 | DT_FP4E2M1, DT_FP4E1M2 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | FP4 is a packed type; two elements share one byte |
| DT_FP4E2M1, DT_FP4E1M2 | DT_BF16 | No rounding involved | Type extension; mode does not affect the result |
| DT_UINT8 | DT_FP16, DT_UINT16 | No rounding involved | Type extension; mode does not affect the result |
| DT_INT8 | DT_FP16, DT_INT16, DT_INT32 | No rounding involved | Type extension; mode does not affect the result |
| DT_INT16 | DT_FP16 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | - |
| DT_INT16 | DT_FP32, DT_UINT32, DT_INT32 | No rounding involved | Type extension; mode does not affect the result |
| DT_INT16 | DT_UINT8 | No rounding involved | Integer narrowing; mode does not affect the result |
| DT_INT32 | DT_FP32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | - |
| DT_INT32 | DT_INT64 | No rounding involved | Type extension; mode does not affect the result |
| DT_INT32 | DT_UINT8, DT_INT16, DT_UINT16 | No rounding involved | Integer narrowing; mode does not affect the result |
| DT_UINT32 | DT_UINT8, DT_UINT16, DT_INT16 | No rounding involved | Integer narrowing; mode does not affect the result |
| DT_INT64 | DT_FP32 | CAST_RINT, CAST_ROUND, CAST_FLOOR, CAST_CEIL, CAST_TRUNC | - |
| DT_INT64 | DT_INT32 | No rounding involved | Integer narrowing; mode does not affect the result |
| DT_FP8E4M3FN, DT_FP8E5M2, DT_HF8 | DT_FP32 | No rounding involved | Type extension; mode does not affect the result |

### Other Constraints

- Both out and src must be UB Tiles. The conversion range is determined by out.valid_shape, and the valid region of src must cover this range. The physical shapes and row strides of the source and destination Tiles may differ, but the elements within the conversion region must correspond one-to-one by the same logical coordinates. The API defines only the conversion result within the valid region of out; do not rely on content outside this region.
- DT_FP32 -> DT_FP16 supports in-place conversion where out and src use exactly the same UB start address. Except for this scenario, the conversion result is not guaranteed when the source and destination storage regions partially or completely overlap. For other data type combinations, allocate non-overlapping UB storage regions for out and src.
- For DT_FP4E2M1 and DT_FP4E1M2, the Tile shape is counted by logical elements, and two adjacent elements are packed into one byte. Therefore, the last dimension of the physical shape must be a multiple of 2. When converting from DT_BF16 to FP4, the last dimension of valid_shape must also be even to ensure that every valid element forms a complete packed byte.
- mode must be determined at compile time and cannot be dynamically selected using runtime Scalar or Tensor values.

## Return Value

None

## Examples

### FP16→FP32 Extension

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

The results are as follows:

<!-- pypto-doc-output:cast:start -->
```bash
Input data src: [[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
Output data out: [[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
```
<!-- pypto-doc-output:cast:end -->

### Rounding Mode

```python
# FP32 -> FP16: Round to the nearest value, with ties rounding to even.
pl.cast(dst_fp16, src_fp32, mode=pl.RoundMode.CAST_RINT)

# FP32 -> INT32: Round toward negative infinity.
# For example, 1.6 is converted to 1, and -1.6 is converted to -2.
pl.cast(dst_int32, src_fp32, mode=pl.RoundMode.CAST_FLOOR)

# FP16 -> FP32: Type extension, no rounding involved.
pl.cast(dst_fp32, src_fp16, mode=pl.RoundMode.CAST_NONE)
```
