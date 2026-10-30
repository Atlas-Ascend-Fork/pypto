# pypto_pro.language.ne

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

将lhs与rhs逐元素比较是否不相等，生成bit-packed掩码。rhs可以是Tile或标量。输出为按位压缩的掩码，通常使用DT_UINT8类型的Tile存储，通常配合[pypto_pro.language.select](../selection/select.md)使用。

## 函数原型

```python
pypto_pro.language.ne(out: Tile, lhs: Tile, rhs: Union[Tile, int, float]) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的操作数，UB Tile，采用行主序布局，保存按位压缩的比较结果。数据类型不限定为DT_UINT8，通常使用DT_UINT8；存储空间须覆盖所有有效元素对应的掩码位。 |
| lhs | 输入 | 左操作数，UB Tile，采用行主序布局。支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_FP32、DT_BF16。 |
| rhs | 输入 | 右操作数，Tile、int或float类型，也支持对应的运行时数值标量表达式。传入Tile时须位于UB，采用行主序布局，数据类型、Shape和有效Shape与lhs一致；传入标量时须能用lhs的元素类型表示。 |

## 返回值说明

无。

## 约束说明

out保存按位压缩的掩码，不能按逐元素0/1数值Tile解释。需要数值结果时，可与[pypto_pro.language.select](../selection/select.md)配合使用。

## 调用示例

将掩码交给pypto_pro.language.select：比较结果为真时选a，否则选b。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def ne_select_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    mask_type = pl.TileType(shape=[64, 64], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ga = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gb = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    gtmp = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    gmask = pl.make_tile_group(type=mask_type, addrs=0x10000, mutex_ids=[4])
    with pl.section_vector():
        ta = ga.current()
        tb = gb.current()
        tout = gout.current()
        tmp = gtmp.current()
        mask = gmask.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.ne(mask, ta, tb)
        pl.select(tout, mask, ta, tb, tmp)
        pl.store(out, tout, [0, 0])
```
