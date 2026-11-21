# pypto_pro.language.partadd

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

Performs element-wise addition on the valid regions of two source tiles. If a position is valid in both source Tiles, the sum of the two source Tiles is output. If a position is valid in only one source tile, the data at the corresponding position in the source Tile is copied.

## Prototype

```python
pypto_pro.language.partadd(out: Tile, src0: Tile, src1: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Target 2D UB Tile, which stores the computation result. The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, and DT_FP32.|
| src0 | Input| First source 2D UB Tile. The supported data types are the same as those of out.|
| src1 | Input| Second source 2D UB Tile. The supported data types are the same as those of out.|

## Return Value

None

## Constraints

- The physical Shapes and data types of out, src0, and src1 must be the same.
- At least one of the two source Tiles must have a valid Shape consistent with that of out; the valid row count and valid column count of the other source Tile must not exceed those of out.
- A smaller valid Shape can be used to specify a partial computation region, but specifying a partial computation region by using a smaller physical Shape is not supported.
- When the valid region of out is empty, the API returns immediately.

## Examples

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def partadd_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1])
    tile_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.set_validshape(cur_a, [64, 128])
        pl.load(cur_a, x, [0, 0])
        pl.set_validshape(cur_b, [32, 128])
        pl.load(cur_b, y, [0, 0])
        pl.set_validshape(cur_c, [64, 128])
        pl.partadd(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
```

Result example:

<!-- pypto-doc-output:partadd:start -->
```bash
Input data x (rows 0, 1, 30, 31, 32, and 33): [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [961 961 961.5 962 962 962 962.5 963 963 963 963.5 964 964 964 964.5 965 ...], [993 993 993.5 994 994 994 994.5 995 995 995 995.5 996 996 996 996.5 997 ...], [1.025000e+03 1.025000e+03 1.026000e+03 1.026000e+03 1.026000e+03 1.026000e+03 1.026000e+03 1.027000e+03 1.027000e+03 1.027000e+03 1.028000e+03 1.028000e+03 1.028000e+03 1.028000e+03 1.028000e+03 1.029000e+03 ...], [1.057000e+03 1.057000e+03 1.058000e+03 1.058000e+03 1.058000e+03 1.058000e+03 1.058000e+03 1.059000e+03 1.059000e+03 1.059000e+03 1.060000e+03 1.060000e+03 1.060000e+03 1.060000e+03 1.060000e+03 1.061000e+03 ...]]
Input data y (rows 0, 1, 30, 31, 32, and 33): [[10 10.5 11 11.5 12 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17 17.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 78 78.5 79 79.5 80 80.5 81 81.5 ...], [1.930000e+03 1.930000e+03 1.931000e+03 1.932000e+03 1.932000e+03 1.932000e+03 1.933000e+03 1.934000e+03 1.934000e+03 1.934000e+03 1.935000e+03 1.936000e+03 1.936000e+03 1.936000e+03 1.937000e+03 1.938000e+03 ...], [1.994000e+03 1.994000e+03 1.995000e+03 1.996000e+03 1.996000e+03 1.996000e+03 1.997000e+03 1.998000e+03 1.998000e+03 1.998000e+03 1.999000e+03 2.000000e+03 2.000000e+03 2.000000e+03 2.001000e+03 2.002000e+03 ...], [2.058000e+03 2.058000e+03 2.060000e+03 2.060000e+03 2.060000e+03 2.060000e+03 2.060000e+03 2.062000e+03 2.062000e+03 2.062000e+03 2.064000e+03 2.064000e+03 2.064000e+03 2.064000e+03 2.064000e+03 2.066000e+03 ...], [2.122000e+03 2.122000e+03 2.124000e+03 2.124000e+03 2.124000e+03 2.124000e+03 2.124000e+03 2.126000e+03 2.126000e+03 2.126000e+03 2.128000e+03 2.128000e+03 2.128000e+03 2.128000e+03 2.128000e+03 2.130000e+03 ...]]
Output data z (rows 0, 1, 30, 31, 32, and 33): [[11 11.75 12.5 13.25 14 14.75 15.5 16.25 17 17.75 18.5 19.25 20 20.75 21.5 22.25 ...], [107 107.75 108.5 109.25 110 110.75 111.5 112.25 113 113.75 114.5 115.25 116 116.75 117.5 118.25 ...], [2.892000e+03 2.892000e+03 2.892000e+03 2.894000e+03 2.894000e+03 2.894000e+03 2.896000e+03 2.896000e+03 2.896000e+03 2.896000e+03 2.898000e+03 2.900000e+03 2.900000e+03 2.900000e+03 2.902000e+03 2.904000e+03 ...], [2.988000e+03 2.988000e+03 2.988000e+03 2.990000e+03 2.990000e+03 2.990000e+03 2.992000e+03 2.992000e+03 2.992000e+03 2.992000e+03 2.994000e+03 2.996000e+03 2.996000e+03 2.996000e+03 2.998000e+03 3.000000e+03 ...], [1.025000e+03 1.025000e+03 1.026000e+03 1.026000e+03 1.026000e+03 1.026000e+03 1.026000e+03 1.027000e+03 1.027000e+03 1.027000e+03 1.028000e+03 1.028000e+03 1.028000e+03 1.028000e+03 1.028000e+03 1.029000e+03 ...], [1.057000e+03 1.057000e+03 1.058000e+03 1.058000e+03 1.058000e+03 1.058000e+03 1.058000e+03 1.059000e+03 1.059000e+03 1.059000e+03 1.060000e+03 1.060000e+03 1.060000e+03 1.060000e+03 1.060000e+03 1.061000e+03 ...]]
```
<!-- pypto-doc-output:partadd:end -->

Lines 0, 1, 30, and 31 are in the valid region of src1, and the output is x+y. Lines 32 and 33 are outside the valid region, and the output remains x.
