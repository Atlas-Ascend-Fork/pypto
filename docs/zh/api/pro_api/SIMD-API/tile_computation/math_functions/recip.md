# pypto_pro.language.recip

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

逐元素计算src的倒数并写入out。浮点类型计算1 / src；DT_INT16和DT_INT32使用整数除法，结果向零截断，例如src为2时结果为0。

## 函数原型

```python
pypto_pro.language.recip(out: Tile, src: Tile) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素倒数结果。<br>数据类型支持：DT_INT16、DT_INT32、DT_FP16、DT_FP32。<br>shape与src保持一致，如果src设置valid_shape，out必须同步设置valid_shape且与src保持一致。<br>支持与src为同一Tile，实现in-place计算。 |
| src | 输入 | 源操作数，Tile类型。<br>数据类型支持：DT_INT16、DT_INT32、DT_FP16、DT_FP32，与out保持一致。<br>元素值不能为0，否则硬件行为不确定，CPU模拟器在debug模式下会触发断言。 |

## 返回值说明

无。

## 约束说明

src的有效元素不得为0。整数类型不保留小数部分，需要浮点倒数时应使用DT_FP16或DT_FP32。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def recip_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                 out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.recip(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:recip:start -->
```bash
输入数据a：[[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [9 9.125 9.25 9.375 9.5 9.625 9.75 9.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [25 25.125 25.25 25.375 25.5 25.625 25.75 25.875 ...], ...]
输出数据out：[[1 0.888889 0.8 0.727273 0.666667 0.615385 0.571429 0.533333 ...], [0.111111 0.109589 0.108108 0.106667 0.105263 0.103896 0.102564 0.101266 ...], [0.058824 0.058394 0.057971 0.057554 0.057143 0.056738 0.056338 0.055944 ...], [0.04 0.039801 0.039604 0.039409 0.039216 0.039024 0.038835 0.038647 ...], ...]
```
<!-- pypto-doc-output:recip:end -->
