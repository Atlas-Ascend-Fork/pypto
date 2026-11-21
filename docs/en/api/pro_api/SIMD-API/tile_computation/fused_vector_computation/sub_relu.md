# pypto_pro.language.sub_relu

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

Performs element-wise subtraction on two source Tiles, applies ReLU activation to the result, and sets elements less than 0 to 0. The formula is out = max(lhs - rhs, 0).

## Prototype

```python
pypto_pro.language.sub_relu(out: Tile, lhs: Tile, rhs: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination UB Tile, which stores the subtraction and ReLU results. DT_FP16, DT_FP32, and DT_INT32 are supported.|
| lhs | Output| First source UB Tile, which is used as the minuend and an intermediate result buffer. The data types are the same as those of out. The original content is overwritten after the call.|
| rhs | Input| Second source UB Tile, which is used as the subtrahend. The data types are the same as those of out.|

## Return Value

None

## Constraints

- The Shapes and data types of out, lhs, and rhs must be the same.
- out may point to the same Tile as lhs or rhs to enable in-place computation.
- This API performs subtraction followed by ReLU activation and uses lhs as the intermediate result; therefore, the original data in lhs will be overwritten.

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sub_relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
                    out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.sub_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

Result example:

<!-- pypto-doc-output:sub_relu:start -->
```bash
Input a: [[2 2.5 3 3.5 4 4.5 5 5.5 ...], [34 34.5 35 35.5 36 36.5 37 37.5 ...], [66 66.5 67 67.5 68 68.5 69 69.5 ...], [98 98.5 99 99.5 100 100.5 101 101.5 ...], ...]
Input b: [[3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [19 19.25 19.5 19.75 20 20.25 20.5 20.75 ...], [35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [51 51.25 51.5 51.75 52 52.25 52.5 52.75 ...], ...]
Output data out: [[0 0 0 0 0 0.25 0.5 0.75 ...], [15 15.25 15.5 15.75 16 16.25 16.5 16.75 ...], [31 31.25 31.5 31.75 32 32.25 32.5 32.75 ...], [47 47.25 47.5 47.75 48 48.25 48.5 48.75 ...], ...]
```
<!-- pypto-doc-output:sub_relu:end -->
