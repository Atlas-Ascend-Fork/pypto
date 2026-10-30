# pypto_pro.language.expand_sub

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

将指定维度的单元素Tile广播到源Tile的shape后执行逐元素减法，即源Tile中每个元素减去广播后对应位置的标量。

## 函数原型

```python
pypto_pro.language.expand_sub(out: Tile, src: Tile, scalar: Tile, *, dim: int=0) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素减法结果。<br>数据类型支持：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。<br>shape与src保持一致。<br>支持与src为同一Tile，实现in-place计算。 |
| src | 输入 | 源操作数，Tile类型。<br>数据类型支持：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。 |
| scalar | 输入 | 广播的具体值，Tile类型，广播到每列或每行做逐元素减法。<br>数据类型支持：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。<br>数据类型须与out、src保持一致。<br>如果src的shape为[M, N]，dim=0时shape为[M, 1]；dim=1时shape为[1, N]。 |
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


@pl.jit(auto_mutex=True)
def row_expand_sub_kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP32],
    y: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                       layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_v, y, [0, 0])
        pl.expand_sub(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_expand_sub:start -->
```bash
输入数据x：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据y：[[1.25], [1.5], [1.75], [2], ...]
输出数据out：[[-4.25 -4.125 -4 -3.875 -3.75 -3.625 -3.5 -3.375 ...], [11.5 11.625 11.75 11.875 12 12.125 12.25 12.375 ...], [27.25 27.375 27.5 27.625 27.75 27.875 28 28.125 ...], [43 43.125 43.25 43.375 43.5 43.625 43.75 43.875 ...], ...]
```
<!-- pypto-doc-output:row_expand_sub:end -->

### dim = 1

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def col_expand_sub_kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP32],
    y: pl.Tensor[[1, 128], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_v, y, [0, 0])
        pl.expand_sub(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_expand_sub:start -->
```bash
输入数据x：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据y：[[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
输出数据out：[[-4.25 -4.375 -4.5 -4.625 -4.75 -4.875 -5 -5.125 ...], [11.75 11.625 11.5 11.375 11.25 11.125 11 10.875 ...], [27.75 27.625 27.5 27.375 27.25 27.125 27 26.875 ...], [43.75 43.625 43.5 43.375 43.25 43.125 43 42.875 ...], ...]
```
<!-- pypto-doc-output:col_expand_sub:end -->
