# pypto_pro.language.quant

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

Quantizes the FP32 source Tile to INT8 or UINT8. scale is the quantization coefficient (reciprocal of the actual scale in common quantization definitions). scale[i, 0] and offset[i, 0] are broadcast by row:

$$
out_{i,j}=\begin{cases}
\operatorname{clamp}_{[-128,127]}\left(\operatorname{roundToEven}\left(src_{i,j}\times scale_{i,0}\right)\right), & mode=\mathrm{SYM} \\
\operatorname{clamp}_{[0,255]}\left(\operatorname{roundToEven}\left(src_{i,j}\times scale_{i,0}+offset_{i,0}\right)\right), & mode=\mathrm{ASYM}
\end{cases}
$$

In asymmetric mode, multiplication and addition are performed in FP32, and then the combined result is rounded once. The product is not rounded before the offset is added.

## Prototype

```python
pypto_pro.language.quant(
    out: Tile,
    src: Tile,
    scale: Tile,
    *,
    mode: QuantMode = pypto_pro.language.QuantMode.SYM,
    offset: Optional[Tile] = None,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, of the Tile type, stored in the UB. The layout must be ND. When mode is SYM, the data type is DT_INT8. When mode is ASYM, the data type is DT_UINT8. The shape and valid_shape must be the same as those of src.|
| src | Input| Source operand, of the Tile type, stored in the UB. The data type is DT_FP32. The layout must be ND. The shape and valid_shape must be the same as those of out.|
| scale | Input| Quantization coefficient, of the Tile type, stored in the UB. The data type is DT_FP32. The layout is ND or DN. The value must be a finite positive number. If src.valid_shape=[M,N], scale.valid_shape must be [M,1]. The number of rows in the shape cannot be less than M, and the number of columns must be 1. The number of valid rows must be the same as that of the output valid rows. The number of physical rows must be 32-byte aligned. In the DT_FP32 scenario, the number of physical rows must be a multiple of 8. scale[i,0] in row i is broadcast to all valid columns in row i of src. This value is directly multiplied by src. Therefore, if the common formula $q=round(x/s)$ is used, $1/s$ should be passed here.|
| mode | Input| Optional. Compile-time enumerated value of [pypto_pro.language.QuantMode](../basic_data_structures/QuantMode.md). It cannot be dynamically specified using a runtime Scalar or Tensor. The value can be SYM (default) or ASYM. The mode also determines the output dtype and whether an offset is required.|
| offset | Input| Quantization zero point, of the Tile type. This parameter is mandatory when mode=ASYM and is recommended to be omitted when mode=SYM. The storage space is UB, the data type is DT_FP32, the layout is ND or DN, and the shape and valid_shape must be the same as those of scale. The number of valid rows must be the same as that of the output valid rows. The number of physical rows must be 32-byte aligned. In the DT_FP32 scenario, the number of physical rows must be a multiple of 8. offset[i,0] in row i is broadcast to the corresponding data row.|

## Constraints

- The computation result uses round-to-nearest (ties to even) and is saturated to the value range of the target data type.

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def quant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP32],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_INT8],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0xA000, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.quant(cur_out, cur_src, cur_scale, mode=pl.QuantMode.SYM)
        pl.store(out, cur_out, [0, 0])
```
