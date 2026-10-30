# pypto_pro.language.exp

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

逐元素计算自然指数e^x。

## 函数原型

```python
pypto_pro.language.exp(out: Tile, src: Tile) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，Tile类型，存放逐元素自然指数结果。<br>数据类型支持DT_FP16、DT_FP32，与src保持一致。shape与src保持一致，src设置valid_shape时，out同步设置valid_shape且与src保持一致。<br>支持与src为同一Tile，实现in-place计算。 |
| src | 输入 | 源操作数，Tile类型，数据类型支持DT_FP16、DT_FP32。 |

## 返回值说明

无。

## 约束说明

无。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def exp_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.exp(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:exp:start -->
```bash
输入数据a：[[-4 -3.875 -3.75 -3.625 -3.5 -3.375 -3.25 -3.125 ...], [4 4.125 4.25 4.375 4.5 4.625 4.75 4.875 ...], [12 12.125 12.25 12.375 12.5 12.625 12.75 12.875 ...], [20 20.125 20.25 20.375 20.5 20.625 20.75 20.875 ...], ...]
输出数据out：[[0.018316 0.020754 0.023518 0.026649 0.030197 0.034218 0.038774 0.043937 ...], [54.598148 61.867809 70.105408 79.439842 90.017128 102.002769 115.584282 130.974152 ...], [1.627548e+05 1.844253e+05 2.089813e+05 2.368068e+05 2.683373e+05 3.040660e+05 3.445519e+05 3.904284e+05 ...], [4.851652e+08 5.497642e+08 6.229645e+08 7.059112e+08 7.999022e+08 9.064079e+08 1.027095e+09 1.163851e+09 ...], ...]
```
<!-- pypto-doc-output:exp:end -->

### in-place用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def exp_inplace_kernel(
    data: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        cur = tile.current()
        pl.load(cur, data, [0, 0])
        pl.exp(cur, cur)
        pl.store(data, cur, [0, 0])
```
