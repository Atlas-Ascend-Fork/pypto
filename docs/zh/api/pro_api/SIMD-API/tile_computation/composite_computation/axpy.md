# pypto_pro.language.axpy

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

向量标量乘加：out[i] = alpha * src[i] + out[i]。将源Tile每个元素乘以标量alpha后累加到目标Tile。

## 函数原型

```python
pypto_pro.language.axpy(out: Tile, src: Tile, alpha: Union[int, float], /) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输入/输出 | 目的操作数，Tile类型，存储空间为UB，数据类型为DT_FP16或DT_FP32，形状和有效形状须与src一致。支持与src同类型计算，也支持src为DT_FP16、out为DT_FP32的拓宽计算，具体类型组合参见[约束说明](#约束说明)。运算前保存累加初值，运算后保存计算结果。 |
| src | 输入 | 源操作数，Tile类型，存储空间为UB，数据类型为DT_FP16或DT_FP32，形状和有效形状须与out一致。 |
| alpha | 输入 | 缩放系数，整型或浮点型常量，或与src元素类型一致的运行时标量表达式，按src的数据类型使用。拓宽计算时仍为DT_FP16，不因out为DT_FP32而改用DT_FP32。 |

## 约束说明

- out和src必须采用行主序排布。
- out和src的数据类型支持以下三种组合，不支持src为DT_FP32、out为DT_FP16的缩窄计算。

| src数据类型 | out数据类型 | alpha数据类型 | 计算方式 |
|---|---|---|---|
| DT_FP16 | DT_FP16 | DT_FP16 | 同类型乘加。 |
| DT_FP32 | DT_FP32 | DT_FP32 | 同类型乘加。 |
| DT_FP16 | DT_FP32 | DT_FP16 | src元素和alpha拓宽为DT_FP32后，与out中的原始值进行乘加，结果保存为DT_FP32。 |

## 返回值说明

无。

## 调用示例

### 计算y = 2.0 × x + y

```python
import pypto_pro.language as pl

ALPHA = 2.0


@pl.jit(auto_mutex=True)
def axpy_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        pl.load(tile_x, x, [0, 0])
        pl.load(tile_y, y, [0, 0])
        pl.axpy(tile_y, tile_x, ALPHA)
        pl.store(y, tile_y, [0, 0])
```

### 运行结果

<!-- pypto-doc-output:axpy:start -->
```bash
输入数据x：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据y原始值：[[4 3.875 3.75 3.625 3.5 3.375 3.25 3.125 ...], [-4 -4.125 -4.25 -4.375 -4.5 -4.625 -4.75 -4.875 ...], [-12 -12.125 -12.25 -12.375 -12.5 -12.625 -12.75 -12.875 ...], [-20 -20.125 -20.25 -20.375 -20.5 -20.625 -20.75 -20.875 ...], ...]
输出数据y：[[6 6.375 6.75 7.125 7.5 7.875 8.25 8.625 ...], [30 30.375 30.75 31.125 31.5 31.875 32.25 32.625 ...], [54 54.375 54.75 55.125 55.5 55.875 56.25 56.625 ...], [78 78.375 78.75 79.125 79.5 79.875 80.25 80.625 ...], ...]
```
<!-- pypto-doc-output:axpy:end -->
