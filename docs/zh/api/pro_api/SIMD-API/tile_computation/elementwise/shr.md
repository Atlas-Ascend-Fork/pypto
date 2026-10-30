# pypto_pro.language.shr

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

将lhs中的每个元素向右移位，并将结果写入out。rhs为Tile时，每个位置使用对应元素指定的移位量；rhs为int或运行时整型标量时，所有元素使用相同的移位量。有符号类型执行算术右移，无符号类型执行逻辑右移。

## 函数原型

```python
pypto_pro.language.shr(out: Tile, lhs: Tile, rhs: Union[Tile, int]) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，UB Tile，采用行主序布局。数据类型与lhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64。Shape和有效Shape与lhs一致。 |
| lhs | 输入 | 被移位操作数，Tile类型。必须位于UB，采用row-major布局。数据类型与out一致。 |
| rhs | 输入 | 移位量，Tile或int类型，也支持运行时整型标量表达式。传入Tile时须为UB、行主序，数据类型、Shape和有效Shape与lhs一致；传入标量时须能用lhs的元素类型表示。移位量须非负且小于lhs的元素位宽。 |

## 返回值说明

无。

## 约束说明

无。

## 调用示例

### Tile-Tile模式

下面是一个完整Kernel：从GM载入两个DT_INT32输入到UB，使用pypto_pro.language.shr逐元素右移后写回GM。Vector Kernel开启auto_mutex，同步由pypto_pro.language.make_tile_group自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def shr_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
               b: pl.Tensor[[64, 64], pl.DT_INT32],
               out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.shr(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

### Tile-Scalar模式

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def shr_scalar_kernel(
    a: pl.Tensor[[64, 64], pl.DT_INT32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.shr(cur_out, cur_a, 1)
        pl.store(out, cur_out, [0, 0])
```
