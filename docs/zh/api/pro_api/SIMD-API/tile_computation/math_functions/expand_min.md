# pypto_pro.language.expand_min

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

将指定维度的单元素Tile广播到源Tile的shape后逐元素取较小值，即源Tile中每个元素与广播后对应位置的标量对比取较小值。

## 函数原型

```python
pypto_pro.language.expand_min(out: Tile, src: Tile, scalar: Tile, *, dim: int=0) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素取较小值结果。<br>数据类型支持：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。<br>shape与src保持一致。 |
| src | 输入 | 源操作数，Tile类型。<br>数据类型支持：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。 |
| scalar | 输入 | 广播的具体值，Tile类型，广播到每列或每行逐元素取较小值。<br>数据类型支持：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。<br>数据类型须与out、src保持一致。<br>如果src的shape为[M, N]，dim=0时shape为[M, 1]；dim=1时shape为[1, N]。 |
| dim | 输入 | 广播方向，0表示沿列方向广播，1表示沿行方向广播。 |

## 返回值说明

无。

## 约束说明

- dim只能为0或1。设src的有效Shape为[M, N]：dim=0时，scalar的有效Shape为[M, 1]；dim=1时为[1, N]。
- out、src和scalar须位于UB，数据类型一致；out与src的Shape和有效Shape一致。

## 调用示例

### dim = 0

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def row_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, 1], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_row = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_row = tile_row.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_row, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_row)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_expand_min:start -->
```bash
输入数据a：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据v：[[1.25], [1.5], [1.75], [2], ...]
输出数据z：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 ...], [1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 ...], [2 2 2 2 2 2 2 2 ...], ...]
```
<!-- pypto-doc-output:row_expand_min:end -->

### dim = 1

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def col_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[1, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_col = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_col = tile_col.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_col, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_col, dim=1)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_expand_min:start -->
```bash
输入数据x：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据y：[[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
输出数据z：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], ...]
```
<!-- pypto-doc-output:col_expand_min:end -->
