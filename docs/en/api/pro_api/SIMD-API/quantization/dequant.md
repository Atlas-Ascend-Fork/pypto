# pypto_pro.language.dequant

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

Dequantizes the INT8 or INT16 source Tile to FP32. scale[i,0] and offset[i,0] are broadcast by row:

$$
out_{i,j}=\left(\operatorname{FP32}(src_{i,j})-offset_{i,0}\right)\times scale_{i,0}
$$

The computation sequence is as follows: Extend and convert the integer source data to FP32, subtract the offset, and multiply the result by the scale. If it is used together with [quant](quant.md), the coefficient passed in the quantization phase is usually the reciprocal of the scale of this API.

## Prototype

```python
pypto_pro.language.dequant(
    out: Tile,
    src: Tile,
    scale: Tile,
    offset: Tile,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, of the Tile type, stored in the UB. The data type is DT_FP32. The layout must be ND. The shape and valid_shape must be the same as those of src.|
| src | Input| Source operand, of the Tile type, stored in the UB. The data type is DT_INT8 or DT_INT16. The layout must be ND. The shape and valid_shape must be the same as those of out.|
| scale | Input| Dequantization coefficient, of the Tile type, stored in the UB. The data type is DT_FP32. The layout is ND or DN. If out.valid_shape=[M,N], scale.valid_shape must be [M,1]. The number of rows in the shape cannot be less than M, and the number of columns must be 1. The number of valid rows must be the same as that of the output valid rows. The number of physical rows must be 32-byte aligned. In the DT_FP32 scenario, the number of physical rows must be a multiple of 8. scale[i,0] in row i is broadcast to all valid columns in row i of the output.|
| offset | Input| Quantization zero point, of the Tile type, stored in UB. The data type is DT_FP32, the layout is ND or DN, and the shape and valid_shape must be the same as those of scale. The number of valid rows must be the same as that of the output valid rows. The number of physical rows must be 32-byte aligned. In the DT_FP32 scenario, the number of physical rows must be a multiple of 8. offset[i,0] in row i is broadcast to the corresponding data row. This parameter cannot be omitted. For symmetric quantization, pass an all-zero Tile.|

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dequant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_INT8],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    offset: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x4000, mutex_ids=[1])
    tile_offset = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                     addrs=0x5000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x6000, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_offset = tile_offset.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.load(cur_offset, offset, [0, 0])
        pl.dequant(cur_out, cur_src, cur_scale, cur_offset)
        pl.store(out, cur_out, [0, 0])
```
