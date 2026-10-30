# pypto_pro.language.gathermask

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

按位模式从源Tile中抽取列到目标Tile，抽取的列由pattern_mode参数决定。

## 函数原型

```python
pypto_pro.language.gathermask(out: Tile, src: Tile, *, pattern_mode: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的UB Tile，数据类型与src一致，保存按模式抽取的元素。容量须覆盖全部被选中元素；有效Shape的设置参见[约束说明](#约束说明)。 |
| src | 输入 | 源操作数，Tile类型，位于UB，行主序，dtype支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_HF8，out的数据类型须与src一致。 |
| pattern_mode | 输入 | 位模式，编译期常量整数，取值1～7，决定抽取哪些列：<br>1：取偶数列，即src[:, 0::2]。<br>2：取奇数列，即src[:, 1::2]。<br>3：每4列取第1列，即src[:, 0::4]。<br>4：每4列取第2列，即src[:, 1::4]。<br>5：每4列取第3列，即src[:, 2::4]。<br>6：每4列取第4列，即src[:, 3::4]。<br>7：取全部列，等价于copy。 |

## 返回值说明

无。

## 约束说明

- pattern_mode必须在编译期确定，取值为1～7。
- out与src的数据类型相同，采用行主序布局，并满足相应Tile的对齐要求。
- 对8、16、32位元素，每行抽取的列数由pattern_mode确定，out须为该结果提供连续存储空间。
- 对64位元素，各源有效行的抽取结果连续拼接写出，不按out的物理列数插入行间填充。设每行选中K个元素、有效行数为R，应提供R * K个元素的连续空间，例如使用有效Shape为[1, R * K]的目标Tile。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p1_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.gathermask(tile_dst, tile_src, pattern_mode=1)
        pl.store(dst, tile_dst, [0, 0])
```

结果示例如下：

<!-- pypto-doc-output:gathermask:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据pattern_mode：1
输出数据dst：[[1 1.5 2 2.5 3 3.5 4 4.5 ...], [33 33.5 34 34.5 35 35.5 36 36.5 ...], [65 65.5 66 66.5 67 67.5 68 68.5 ...], [97 97.5 98 98.5 99 99.5 100 100.5 ...], ...]
```
<!-- pypto-doc-output:gathermask:end -->

### 抽取奇数列

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p2_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    source_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    output_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    gsrc = pl.make_tile_group(type=source_type, addrs=0x0000, mutex_ids=[0])
    gout = pl.make_tile_group(type=output_type, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tsrc = gsrc.current()
        tout = gout.current()
        pl.load(tsrc, src, [0, 0])
        pl.gathermask(tout, tsrc, pattern_mode=2)
        pl.store(dst, tout, [0, 0])
```

### 复制全部列

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p7_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    source_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    output_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    gsrc = pl.make_tile_group(type=source_type, addrs=0x0000, mutex_ids=[0])
    gout = pl.make_tile_group(type=output_type, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tsrc = gsrc.current()
        tout = gout.current()
        pl.load(tsrc, src, [0, 0])
        pl.gathermask(tout, tsrc, pattern_mode=7)
        pl.store(dst, tout, [0, 0])
```
