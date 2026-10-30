# pypto_pro.language.mul_add_dst

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

对lhs和rhs执行逐元素乘法，再将乘法结果与out中的原始数据执行逐元素加法，并将最终结果写入out。计算公式为out = lhs * rhs + out。

## 函数原型

```python
pypto_pro.language.mul_add_dst(out: Tile, lhs: Tile, rhs: Tile) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，UB Tile，支持DT_FP16、DT_FP32。调用前的值参与加法，调用后保存计算结果。 |
| lhs | 输入 | 参与乘法的源UB Tile，数据类型、Shape和有效Shape与out一致。与out使用互不重叠的区域时，调用后保留原始数据。 |
| rhs | 输入 | 源UB Tile，数据类型、Shape和有效Shape与out一致。 |

## 返回值说明

无。

## 约束说明

- out、lhs和rhs的数据类型必须相同，支持DT_FP16、DT_FP32。
- 三个Tile须采用相同的行主序布局，有效Shape一致，并满足Tile布局的对齐要求。
- out中的原始数据参与计算，调用前必须初始化。
- 为保留源数据，源Tile与out应使用互不重叠的存储区域。

## 调用示例

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def mul_add_dst_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.load(cur_c, z, [0, 0])
        pl.mul_add_dst(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
```

结果示例如下：

<!-- pypto-doc-output:mul_add_dst:start -->
```bash
输入数据x：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据y：[[3 2.875 2.75 2.625 2.5 2.375 2.25 2.125 ...], [-13 -13.125 -13.25 -13.375 -13.5 -13.625 -13.75 -13.875 ...], [-29 -29.125 -29.25 -29.375 -29.5 -29.625 -29.75 -29.875 ...], [-45 -45.125 -45.25 -45.375 -45.5 -45.625 -45.75 -45.875 ...], ...]
输入数据z原始值：[[-2 -1.9375 -1.875 -1.8125 -1.75 -1.6875 -1.625 -1.5625 ...], [6 6.0625 6.125 6.1875 6.25 6.3125 6.375 6.4375 ...], [14 14.0625 14.125 14.1875 14.25 14.3125 14.375 14.4375 ...], [22 22.0625 22.125 22.1875 22.25 22.3125 22.375 22.4375 ...], ...]
输出数据z：[[1 1.65625 2.25 2.78125 3.25 3.65625 4 4.28125 ...], [-423 -430.5 -438 -445.25 -452.75 -460.5 -468 -475.75 ...], [-1.871000e+03 -1.886000e+03 -1.902000e+03 -1.917000e+03 -1.933000e+03 -1.949000e+03 -1.964000e+03 -1.980000e+03 ...], [-4.344000e+03 -4.364000e+03 -4.388000e+03 -4.412000e+03 -4.436000e+03 -4.460000e+03 -4.484000e+03 -4.508000e+03 ...], ...]
```
<!-- pypto-doc-output:mul_add_dst:end -->
