# pypto_pro.language.mul_cast

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

Performs element-wise multiplication on the valid regions of the two source Tiles, and then converts the intermediate result to the destination data type. This API does not perform ReLU.

$$
out_{i,j}=\operatorname{cast}_{mode}\left(lhs_{i,j}\times rhs_{i,j}\right)
$$

During computation, lhs also serves as the intermediate result buffer for multiplication. The multiplication result is written back to lhs, then the data type is converted, and the result is written to out. Therefore, the original content of lhs is overwritten after the call is complete.

## Prototype

```python
pypto_pro.language.mul_cast(
    out: Tile,
    lhs: Tile,
    rhs: Tile,
    *,
    target_type: DType,
    mode: RoundMode=RoundMode.CAST_ROUND,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| UB, RowMajor Tile; its physical shape, valid_shape, and layout must be consistent with those of lhs and rhs. Its dtype must be consistent with target_type; see the table below for supported target types.|
| lhs | Output| UB, RowMajor Tile; its dtype must be consistent with that of rhs and meet the data type combinations supported by this API. Its physical shape, valid_shape, and layout must be consistent with those of rhs and out. After the API is called, the content changes to the intermediate result of the multiplication.|
| rhs | Input| UB, RowMajor Tile; its dtype, physical shape, valid_shape, and layout must be consistent with those of lhs.|
| target_type | Input| A compile-time [pypto_pro.language.DataType](../../basic_data_structures/DataType.md) enumeration value, which must be equal to out.dtype. This parameter does not reallocate or change the type of out; any inconsistency between the two is an unsupported usage.|
| mode | Input| (Optional) Compile-time [pypto_pro.language.RoundMode](../../basic_data_structures/RoundMode.md) enumeration value. The default value is pypto_pro.language.RoundMode.CAST_ROUND. The supported conversion paths and rounding modes are the same as those of [pypto_pro.language.cast](../type_conversion/cast.md).|

### Supported Data Type Conversions

The multiplication is performed using the data type of lhs/rhs, and then the intermediate result is converted to target_type. The following data type combinations are supported. Combinations that are not listed are not supported.

| lhs/rhs dtype | target_type (also out.dtype)|
|---|---|
| DT_FP16 | DT_FP32, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, and DT_HF8|
| DT_FP32 | DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_INT64, DT_FP8E4M3FN, DT_FP8E5M2, and DT_HF8|
| DT_BF16 | DT_FP32, DT_INT32, DT_FP16, DT_FP4E2M1, and DT_FP4E1M2|
| DT_INT16 | DT_FP16, DT_FP32, DT_UINT32, DT_INT32, and DT_UINT8|
| DT_INT32 | DT_FP32, DT_INT64, DT_UINT8, DT_INT16, and DT_UINT16|
| DT_UINT32 | DT_UINT8, DT_UINT16, and DT_INT16|
| DT_INT64 | DT_FP32 and DT_INT32|

The rounding mode is interpreted based on the conversion path from the intermediate type to the target type. In integer narrowing and lossless widening paths, mode has no effect on the result. DT_FP32 → DT_FP16 supports CAST_ODD; DT_FP16/DT_FP32 → DT_HF8 always uses CAST_ROUND. For details about the complete rules, see [Constraints on cast](../type_conversion/cast.md#constraints).

## Return Value

None. The computation result is written to out.

## Constraints

1. lhs will be overwritten, and its pre-call value must not be relied on afterward. rhs and out must use UB regions that do not overlap with lhs; this API does not guarantee results for arbitrary address overlaps.
2. The API defines results only within the valid region specified by valid_shape; content outside the valid region must not be relied on.
3. target_type and mode must be determined at the compile time and cannot be dynamically selected using runtime Scalars or Tensors.

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mul_cast_kernel(a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16],
                    out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt_in = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_in, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.mul_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])
```

Result example:

<!-- pypto-doc-output:mul_cast:start -->
```bash
Input a: [[-2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 ...], [14 14.25 14.5 14.75 15 15.25 15.5 15.75 ...], [30 30.25 30.5 30.75 31 31.25 31.5 31.75 ...], [46 46.25 46.5 46.75 47 47.25 47.5 47.75 ...], ...]
Input b: [[3 2.875 2.75 2.625 2.5 2.375 2.25 2.125 ...], [-5 -5.125 -5.25 -5.375 -5.5 -5.625 -5.75 -5.875 ...], [-13 -13.125 -13.25 -13.375 -13.5 -13.625 -13.75 -13.875 ...], [-21 -21.125 -21.25 -21.375 -21.5 -21.625 -21.75 -21.875 ...], ...]
Output data: [[-6 -5.03125 -4.125 -3.28125 -2.5 -1.78125 -1.125 -0.53125 ...], [-70 -73 -76.125 -79.25 -82.5 -85.75 -89.125 -92.5 ...], [-390 -397 -404 -411.25 -418.5 -425.75 -433 -440.5 ...], [-966 -977 -988 -999.5 -1.010500e+03 -1.022000e+03 -1.033000e+03 -1.045000e+03 ...], ...]
```
<!-- pypto-doc-output:mul_cast:end -->
