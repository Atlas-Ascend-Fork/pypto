# pypto_pro.language.minimum

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

minimum同时支持逐元素取较小值和按维度取最小值归约。是否传入dim决定调用模式。

- **Tile-Tile模式**：比较lhs和rhs对应位置的元素，将较小值写入out。
- **Tile-Scalar模式**：比较lhs中的每个元素与rhs标量，将较小值写入out。
- **归约模式**：对lhs沿dim指定的维度取最小值，将结果写入out。

## 函数原型

```python
pypto_pro.language.minimum(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, int, float],
    *,
    dim: Optional[int]=None,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素计算结果或归约结果。逐元素模式下数据类型与lhs一致，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32；归约模式下数据类型与lhs一致，dim=0时shape为[行数, 1]，dim=1时shape为[1, 列数]。 |
| lhs | 输入 | 源Tile。逐元素模式下为左操作数，支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16；归约模式下须为二维Tile，支持的类型参见[约束说明](#约束说明)。 |
| rhs | 输入 | Tile、int或float类型，也支持对应的运行时数值标量表达式。逐元素模式下为右操作数，Tile-Tile时其数据类型、Shape和有效Shape与lhs一致；归约模式下须提供临时Tile，数据类型与lhs一致。 |
| dim | 输入 | 可选，归约维度。未传入时执行逐元素计算；传入0时沿最后一维归约；传入1时沿第一维归约。 |

## 返回值说明

无。

## 约束说明

- 逐元素模式下，out、lhs以及Tile类型的rhs须位于UB，数据类型、Shape和有效Shape一致。
- 归约模式下，dim只能为0或1，out的数据类型与lhs一致；dim=0时输出Shape为[M, 1]，dim=1时为[1, N]。

| 归约维度 | lhs支持的数据类型 |
|---|---|
| dim=0 | DT_FP16、DT_FP32、DT_INT32、DT_INT64、DT_UINT64、DT_INT16、DT_INT8、DT_UINT8 |
| dim=1 | DT_FP16、DT_FP32、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_BF16 |

## 调用示例

### Tile-Tile模式

下面是一个完整Kernel：从GM载入两个DT_FP32输入到UB，使用pypto_pro.language.minimum逐元素取较小值后写回GM。Vector Kernel开启auto_mutex，同步由pypto_pro.language.make_tile_group自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def minimum_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
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
        pl.minimum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

结果示例如下。

<!-- pypto-doc-output:minimum:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据b：[[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
输出数据out：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
```
<!-- pypto-doc-output:minimum:end -->

### Tile-Scalar模式

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def minimum_scalar_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.minimum(cur_out, cur_a, 100.0)
        pl.store(out, cur_out, [0, 0])
```

### 归约模式

#### dim = 0结果示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.minimum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_min:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[-8], [24], [56], [88], ...]
```
<!-- pypto-doc-output:row_min:end -->

#### dim = 1结果示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def col_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.minimum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_min:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...]]
```
<!-- pypto-doc-output:col_min:end -->
