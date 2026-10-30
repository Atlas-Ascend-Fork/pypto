# pypto_pro.language.gather

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

支持按索引聚合和按比较条件收集索引两种模式。

- **索引模式**：未显式传入cmp_mode时，根据indices_or_k_value中的扁平元素索引读取src，并写入out。与[pypto_pro.language.scatter](scatter.md)对应。
- **比较模式**：显式传入cmp_mode时，按src类型对应的规则与整数阈值比较，把满足条件的元素索引写入out，并将每行写出的字节数写入计数Tile。可用于TopK等场景的索引筛选，类型及数值约束参见[约束说明](#约束说明)。

## 函数原型

```python
pypto_pro.language.gather(
    out: Tile,
    src: Tile,
    indices_or_k_value: Tile,
    tmp_or_cdst: Optional[Tile]=None,
    tmp: Optional[Tile]=None,
    *,
    cmp_mode: int=0,
    offset: int=0,
) -> None
```

索引模式使用前三个位置参数，或再提供第四个临时Tile；不传cmp_mode。比较模式须提供五个位置参数，并显式传入cmp_mode。

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| out | 输出 | 目的UB Tile。索引模式下，数据类型与src相同，Shape和有效Shape须与索引结果相匹配；比较模式下使用DT_INT32或DT_UINT32，保存选中的索引，容量须覆盖可能选中的全部元素。 |
| src | 输入 | 源UB Tile。索引模式读取其扁平元素；比较模式支持DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_FP16、DT_FP32，比较规则及模式限制参见[约束说明](#约束说明)。 |
| indices_or_k_value | 输入 | 索引模式下为索引Tile，支持DT_INT16、DT_UINT16、DT_INT32、DT_UINT32，索引以元素为单位；64位源数据须使用32位索引。比较模式下为整数阈值Tile，必须使用DT_UINT16或DT_UINT32，不要求与src同位宽，也不使用浮点Tile。阈值按源有效行的顺序连续存放，第i个元素对应src的第i行。 |
| tmp_or_cdst | 输出 | 可选，默认None。索引模式下为兼容性临时Tile，可省略；比较模式下必须传入输出计数Tile，保存每行写入out的字节数，使用DT_INT32或DT_UINT32。32位索引的有效元素个数等于该字节数除以4。 |
| tmp | 输入 | 可选，默认None。索引模式不传此参数；比较模式必须作为第五个位置参数提供兼容性临时Tile。 |
| cmp_mode | 输入 | 比较模式控制参数，编译期整数。显式传入0表示相等比较，4表示大于比较；当前仅支持这两个值。索引模式须省略此参数，显式传入0不表示关闭比较。 |
| offset | 输入 | 可选，编译期整数，默认0。仅用于比较模式，加到收集到的索引值上，不改变src的读取位置，也不对索引模式的输入索引施加偏移。 |

## 返回值说明

无。

## 约束说明

- 所有Tile须位于UB，并按相应Tile布局要求分配存储空间；不同用途的Tile应使用互不重叠的区域。
- 索引模式下，out与src的数据类型必须一致，索引Tile的有效区域须覆盖out的有效区域。每个输入索引必须指向src中可读取的元素。
- 比较模式下，src、阈值、out和计数Tile采用行主序布局。必须提供计数Tile与第五个临时Tile；不得把索引模式的四参数调用与cmp_mode组合使用。
- 比较模式下，仅out每行前“计数字节数 / 4”个索引有效。计数Tile记录字节数，不是元素个数。

### 比较模式的类型与数值约束

- 比较模式用于非负数值与整数阈值的筛选。源为有符号整型时，不应把负值按普通有符号数的大小关系解释。
- src为DT_UINT16时，当前仅使用相等筛选，即cmp_mode=0；当前类型分派没有为该类型提供独立的大于比较路径，不能用cmp_mode=4获得大于筛选结果。
- src为DT_FP32时，先按底层ROUND_R规则转换为DT_INT32，再将转换结果按DT_UINT32解释，与整数阈值比较。因此，它不等价于对原始浮点数直接执行相等或大小比较。例如src元素为20.25、阈值为20时，转换后为20，在cmp_mode=0下会被选中。
- DT_FP32源须为有限值，转换后的值应位于DT_INT32的非负范围内；NaN、无穷大或越界输入不适用于上述筛选语义。
- src为DT_FP16时，整数阈值经过16位有符号整数转换后与提升至DT_FP32的源数据比较；阈值应使用[0, 32767]范围内的整数，不应传入DT_FP16位模式代替整数阈值。

## 调用示例

### 索引模式

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gather_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    indices: pl.Tensor[[64, 128], pl.DT_INT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt_fp16 = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_int32 = pl.TileType(shape=[64, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_fp16, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_int32, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt_int32, addrs=0xC000, mutex_ids=[2])
    tile_dst = pl.make_tile_group(type=tt_fp16, addrs=0x14000, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_tmp = tile_tmp.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, indices, [0, 0])
        pl.gather(cur_dst, cur_src, cur_idx, cur_tmp)
        pl.store(dst, cur_dst, [0, 0])
```

结果示例如下：

<!-- pypto-doc-output:gather:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 67 67.25 67.5 67.75 68 68.25 68.5 68.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 99 99.25 99.5 99.75 100 100.25 100.5 100.75 ...], ...]
输入数据indices：[[127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112 ...], [255 254 253 252 251 250 249 248 247 246 245 244 243 242 241 240 ...], [383 382 381 380 379 378 377 376 375 374 373 372 371 370 369 368 ...], [511 510 509 508 507 506 505 504 503 502 501 500 499 498 497 496 ...], ...]
输出数据dst：[[32.75 32.5 32.25 32 31.75 31.5 31.25 31 30.75 30.5 30.25 30 29.75 29.5 29.25 29 ...], [64.75 64.5 64.25 64 63.75 63.5 63.25 63 62.75 62.5 62.25 62 61.75 61.5 61.25 61 ...], [96.75 96.5 96.25 96 95.75 95.5 95.25 95 94.75 94.5 94.25 94 93.75 93.5 93.25 93 ...], [128.75 128.5 128.25 128 127.75 127.5 127.25 127 126.75 126.5 126.25 126 125.75 125.5 125.25 125 ...], ...]
```
<!-- pypto-doc-output:gather:end -->

### 比较模式

下面的Kernel收集大于20的元素位置，并为输出索引加16。indices_out中只有byte_count_out[0, 0] / 4个索引有效；其余位置不应作为聚合结果使用。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gather_compare_kernel(
    src: pl.Tensor[[1, 128], pl.DT_INT16],
    indices_out: pl.Tensor[[1, 128], pl.DT_UINT32],
    byte_count_out: pl.Tensor[[1, 16], pl.DT_UINT32],
):
    src_type = pl.TileType(shape=[1, 128], dtype=pl.DT_INT16, target_memory=pl.MemorySpace.Vec)
    key_type = pl.TileType(shape=[1, 16], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec)
    index_type = pl.TileType(shape=[1, 128], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    count_type = pl.TileType(shape=[1, 16], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tmp_type = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    gsrc = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    gkey = pl.make_tile_group(type=key_type, addrs=0x0100, mutex_ids=[1])
    gindices = pl.make_tile_group(type=index_type, addrs=0x0200, mutex_ids=[2])
    gcount = pl.make_tile_group(type=count_type, addrs=0x0400, mutex_ids=[3])
    gtmp = pl.make_tile_group(type=tmp_type, addrs=0x0500, mutex_ids=[4])
    with pl.section_vector():
        tsrc = gsrc.current()
        key = gkey.current()
        indices = gindices.current()
        count = gcount.current()
        tmp = gtmp.current()
        pl.load(tsrc, src, [0, 0])
        pl.expands(key, 20)
        pl.expands(indices, 0)
        pl.expands(count, 0)
        # 收集大于20的元素索引；offset加到收集到的索引值上。
        pl.gather(indices, tsrc, key, count, tmp, cmp_mode=4, offset=16)
        pl.store(indices_out, indices, [0, 0])
        pl.store(byte_count_out, count, [0, 0])
```

若src前8个元素为[0, 10, 20, 30, 20, 10, 40, 20]且其余元素均为0，则有效输出索引为[19, 22]，计数字节数为8。
