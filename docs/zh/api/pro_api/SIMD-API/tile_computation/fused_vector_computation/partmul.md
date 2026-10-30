# pypto_pro.language.partmul

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

对两个源Tile的有效区域执行逐元素乘法。某一位置在两个源Tile中均有效时，输出两者的乘积；仅在一个源Tile中有效时，复制该源Tile对应位置的数据。

## 函数原型

```python
pypto_pro.language.partmul(out: Tile, src0: Tile, src1: Tile) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的二维UB Tile，保存部分区域乘法结果。支持DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。 |
| src0 | 输入 | 第一个源二维UB Tile。支持的数据类型与out相同。 |
| src1 | 输入 | 第二个源二维UB Tile。支持的数据类型与out相同。 |

## 返回值说明

无。

## 约束说明

- out、src0和src1的物理Shape和数据类型须一致。
- 两个源Tile中至少一个的有效Shape须与out的有效Shape一致，另一个源Tile的有效行数和有效列数均不得超过out。
- 通过较小的有效Shape指定部分计算区域，不支持使用较小的物理Shape指定部分计算区域。
- out的有效区域为空时，接口直接返回。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def partmul_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16],
                   y: pl.Tensor[[64, 128], pl.DT_FP16],
                   out: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gx = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        tx, ty, tout = gx.current(), gy.current(), gout.current()
        pl.set_validshape(tx, [64, 128])
        pl.load(tx, x, [0, 0])
        pl.set_validshape(ty, [32, 128])
        pl.load(ty, y, [0, 0])
        pl.set_validshape(tout, [64, 128])
        pl.partmul(tout, tx, ty)
        pl.store(out, tout, [0, 0])
```

前32行计算x * y，后32行只有x有效，因此复制x。
