# pypto_pro.language.gt

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

将lhs与rhs逐元素比较，当lhs大于rhs时生成真值，结果以bit-packed掩码输出。rhs可以是Tile或标量。输出为按位压缩的掩码，通常使用DT_UINT8类型的Tile存储，通常配合[pypto_pro.language.select](../selection/select.md)使用。

## 函数原型

```python
pypto_pro.language.gt(out: Tile, lhs: Tile, rhs: Union[Tile, int, float]) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，UB Tile，采用行主序布局，保存按位压缩的比较结果。数据类型不限定为DT_UINT8，通常使用DT_UINT8；存储空间须覆盖所有有效元素对应的掩码位。 |
| lhs | 输入 | 左操作数，UB Tile，采用行主序布局。支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。 |
| rhs | 输入 | 右操作数，Tile、int或float类型，也支持对应的运行时数值标量表达式。传入Tile时须位于UB，采用行主序布局，数据类型、Shape和有效Shape与lhs一致；传入标量时须能用lhs的元素类型表示。 |

## 返回值说明

无。

## 约束说明

out保存按位压缩的掩码，不能按逐元素0/1数值Tile解释。需要数值结果时，可与[pypto_pro.language.select](../selection/select.md)配合使用。

## 调用示例

### Tile-Tile比较

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gt_select_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tmp_vec_group = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    mask_vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000, mutex_ids=[4])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        tmp_vec = tmp_vec_group.current()
        mask_vec = mask_vec_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.gt(mask_vec, tile_a, tile_b)
        pl.select(tile_out, mask_vec, tile_a, tile_b, tmp_vec)
        pl.store(out, tile_out, [0, 0])
```

### Tile-Tile比较结果

<!-- pypto-doc-output:comparison_tile:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据b：[[8 7.875 7.75 7.625 7.5 7.375 7.25 7.125 ...], [0 -0.125 -0.25 -0.375 -0.5 -0.625 -0.75 -0.875 ...], [-8 -8.125 -8.25 -8.375 -8.5 -8.625 -8.75 -8.875 ...], [-16 -16.125 -16.25 -16.375 -16.5 -16.625 -16.75 -16.875 ...], ...]
输出数据out：[[8 7.875 7.75 7.625 7.5 7.375 7.25 7.125 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
```
<!-- pypto-doc-output:comparison_tile:end -->

### Tile-Scalar比较

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def scalar_gt_select_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    b: pl.Tensor[[64, 128], pl.DT_FP32],
    mask_in: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tt32 = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt32, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt32, addrs=0x8000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt32, addrs=0x10000, mutex_ids=[2])
    tmp_vec_group = pl.make_tile_group(type=tt32, addrs=0x18000, mutex_ids=[3])
    mask_fp16_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x20000, mutex_ids=[4])
    mask_vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x24000, mutex_ids=[5])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        tmp_vec = tmp_vec_group.current()
        mask_fp16 = mask_fp16_group.current()
        mask_vec = mask_vec_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.load(mask_fp16, mask_in, [0, 0])
        # mask_fp16 > 0，生成按位压缩的掩码mask_vec
        pl.gt(mask_vec, mask_fp16, 0.0)
        # 谓词为真取 lhs(=a)，否则取 rhs(=b)
        pl.select(tile_out, mask_vec, tile_a, tile_b, tmp_vec)
        pl.store(out, tile_out, [0, 0])
```

### Tile-Scalar比较结果

<!-- pypto-doc-output:comparison_scalar:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据b：[[8 7.875 7.75 7.625 7.5 7.375 7.25 7.125 ...], [-8 -8.125 -8.25 -8.375 -8.5 -8.625 -8.75 -8.875 ...], [-24 -24.125 -24.25 -24.375 -24.5 -24.625 -24.75 -24.875 ...], [-40 -40.125 -40.25 -40.375 -40.5 -40.625 -40.75 -40.875 ...], ...]
输入数据mask：[[1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], ...]
输出数据out：[[1 7.875 1.5 7.625 2 7.375 2.5 7.125 ...], [33 -8.125 33.5 -8.375 34 -8.625 34.5 -8.875 ...], [65 -24.125 65.5 -24.375 66 -24.625 66.5 -24.875 ...], [97 -40.125 97.5 -40.375 98 -40.625 98.5 -40.875 ...], ...]
```
<!-- pypto-doc-output:comparison_scalar:end -->
