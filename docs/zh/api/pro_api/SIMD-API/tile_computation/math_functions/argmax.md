# pypto_pro.language.argmax

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

沿指定维度获取源Tile的最大元素索引并写入目的Tile。

## 函数原型

```python
pypto_pro.language.argmax(out: Tile, src: Tile, tmp: Tile, *, dim: int=0) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放每行或者每列最大元素的索引值，支持的数据类型详见[约束说明](#约束说明)。<br>如果src的shape为[M, N]，dim=0时shape为[M, 1]；dim=1时shape为[1, N]。 |
| src | 输入 | 源操作数，Tile类型，支持的数据类型详见[约束说明](#约束说明)。 |
| tmp | 输入 | 兼容性临时Tile，当前调用仍须提供。建议使用与src相同的数据类型、Shape和有效Shape。 |
| dim | 输入 | 查找维度，0表示获取每行最大值对应列索引，1表示获取每列最大值对应行索引。 |

## 返回值说明

无。

## 约束说明

- dim只能为0或1，分别查找每行的列索引、每列的行索引。
- src仅支持DT_FP16、DT_FP32；out仅支持DT_INT32、DT_UINT32，两种维度取值下的类型范围相同。
- 所有Tile须位于UB。src的有效区域须非空，out的有效Shape应与归约结果一致。

## 调用示例

### dim = 0

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def row_argmax_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, 1], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmax(cur_out, cur_a, cur_tmp)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_argmax:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据z：[[127], [127], [127], [127], ...]
```
<!-- pypto-doc-output:row_argmax:end -->

### dim = 1

```python
import pypto_pro.language as pl


M, N = 64, 128

@pl.jit(auto_mutex=True)
def col_argmax_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[1, N], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmax(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_argmax:start -->
```bash
输入数据x：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据z：[[63 63 63 63 63 63 63 63 ...]]
```
<!-- pypto-doc-output:col_argmax:end -->
