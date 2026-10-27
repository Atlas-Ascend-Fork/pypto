# pypto_pro.language.mrgsort2

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

多路归并排序：将2到4个已经按value降序排列的源列表归并为一个降序输出。每个源Tile保存固定8字节的value-index记录，记录布局与[sort32](sort32.md)输出一致：FP32为[value_f32(4Byte), index_u32(4Byte)]，FP16为[value_f16(2Byte), padding(2Byte, 0), index_u32(4Byte)]。

底层vmrgsort4先把归并结果写入tmp，再将dst.valid_shape指定的前缀复制到dst。因此，dst可以小于全部输入之和，但tmp仍须容纳完整的输入列表总容量。

exhausted控制硬件在某个输入列表耗尽时是否暂停归并，用于多步归并中处理长度不一致的源；它是整次调用的控制位，不用于指定某一个源。底层会记录每个列表实际消费的元素数，但当前Pro接口不返回这些计数。

## 函数原型

```python
pypto_pro.language.mrgsort2(
    src0: Tile,
    src1: Tile,
    dst: Tile,
    tmp: Tile,
    *args,
    exhausted: bool=False,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| src0 | 输入 | UB、RowMajor、单行Tile；dtype为DT_FP16或DT_FP32。内容须为按value降序排列的8字节value-index记录。 |
| src1 | 输入 | UB、RowMajor、单行Tile；dtype为DT_FP16或DT_FP32。内容须为按value降序排列的8字节value-index记录。 |
| dst | 输出 | UB、RowMajor、单行Tile；dtype与所有源一致。输出列数由dst.valid_shape[1]控制，可小于全部源的总列数，但不得大于可归并数据量。 |
| tmp | 输出 | UB、RowMajor、单行Tile，dtype与所有源一致；物理列数至少为所有源物理列数之和，公式见下文。 |
| *args | 输入 | 可选，额外的已排序源Tile（支持3路或4路归并），使源总数为2～4个；每个Tile须满足与src0/src1相同的类型、布局和有序性要求。 |
| exhausted | 输入 | 可选，编译期bool，默认False；True表示任一输入列表耗尽时停止当前硬件归并。当前Pro接口不返回各列表的实际消费数量。 |

### tmp尺寸

设参与归并的源Tile物理列数依次为src0.shape[1]、src1.shape[1]等，则：

$$
tmp.shape[1]\ge\sum_k src_k.shape[1]
$$

即：

- 2路：tmp.shape[1] >= src0.shape[1] + src1.shape[1]；
- 3路：再加src2.shape[1]；
- 4路：再加src2.shape[1] + src3.shape[1]。

tmp按Tile存储元素计数，不能按value-index记录数计算。FP32每条记录占2列，FP16每条记录占4列。

## 返回值说明

无返回值。归并结果写入dst；各输入列表的实际消费数量不返回。

## 约束说明

1. 所有源、dst和tmp必须为UB、RowMajor、单行Tile，dtype完全一致，并使用互不重叠且按32字节对齐的UB区域。
2. 每个源Tile的有效列数必须是完整8字节记录的整数倍：FP32为2的倍数，FP16为4的倍数。每个源列表包含的8字节记录数不得超过4095，即FP32的有效列数不得超过8190，FP16的有效列数不得超过16380。
3. 每个源列表在调用前必须已经按value降序排列。接口不检查输入有序性；输入无序时输出不保证有序。
4. 所有Tile的地址区间须满足目标平台的UB容量检查。SIMD Tile地址按目标平台的物理容量检查；含SIMT的混合Kernel还受静态和动态用户UB额度限制，不能用统一的248KiB上限代替这两类条件。临时Tile的容量仍须满足本页的tmp尺寸要求。
5. exhausted=True时，Pro接口不会向调用者暴露底层MrgSortExecutedNumList。若算法需要精确获知各源消费数量，当前接口不能提供该信息。
6. 结果按value降序排列；value相同时，index较小的记录排在前面。NaN的相对顺序由底层浮点比较行为决定。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mrgsort2_kernel(
    src0_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src1_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    sorted_out: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_tmp = pl.TileType(shape=[1, 512], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_src0_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_src1_group = pl.make_tile_group(type=tt, addrs=0x0400, mutex_ids=[1])
    tile_dst_group = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[2])
    tile_tmp_group = pl.make_tile_group(type=tt_tmp, addrs=0x0C00, mutex_ids=[3])
    with pl.section_vector():
        tile_src0 = tile_src0_group.current()
        tile_src1 = tile_src1_group.current()
        tile_dst = tile_dst_group.current()
        tile_tmp = tile_tmp_group.current()
        pl.load(tile_src0, src0_tensor, [0, 0])
        pl.load(tile_src1, src1_tensor, [0, 0])
        pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, exhausted=False)
        pl.store(sorted_out, tile_dst, [0, 0])
```

### 三路归并

各输入均须为已降序排列的value-index记录。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mrgsort2_3way_kernel(
    src0_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src1_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src2_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tmp_type = pl.TileType(shape=[1, 768], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gsrc0 = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gsrc1 = pl.make_tile_group(type=tt, addrs=0x0400, mutex_ids=[1])
    gsrc2 = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[2])
    gout = pl.make_tile_group(type=tt, addrs=0x0C00, mutex_ids=[3])
    gtmp = pl.make_tile_group(type=tmp_type, addrs=0x1000, mutex_ids=[4])
    with pl.section_vector():
        src0 = gsrc0.current()
        src1 = gsrc1.current()
        src2 = gsrc2.current()
        dst = gout.current()
        tmp = gtmp.current()
        pl.load(src0, src0_tensor, [0, 0])
        pl.load(src1, src1_tensor, [0, 0])
        pl.load(src2, src2_tensor, [0, 0])
        pl.mrgsort2(src0, src1, dst, tmp, src2, exhausted=False)
        pl.store(out, dst, [0, 0])
```

### 四路归并

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mrgsort2_4way_kernel(
    src0_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src1_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src2_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src3_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tmp_type = pl.TileType(shape=[1, 1024], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    gsrc0 = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gsrc1 = pl.make_tile_group(type=tt, addrs=0x0400, mutex_ids=[1])
    gsrc2 = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[2])
    gsrc3 = pl.make_tile_group(type=tt, addrs=0x0C00, mutex_ids=[3])
    gout = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[4])
    gtmp = pl.make_tile_group(type=tmp_type, addrs=0x1400, mutex_ids=[5])
    with pl.section_vector():
        src0 = gsrc0.current()
        src1 = gsrc1.current()
        src2 = gsrc2.current()
        src3 = gsrc3.current()
        dst = gout.current()
        tmp = gtmp.current()
        pl.load(src0, src0_tensor, [0, 0])
        pl.load(src1, src1_tensor, [0, 0])
        pl.load(src2, src2_tensor, [0, 0])
        pl.load(src3, src3_tensor, [0, 0])
        pl.mrgsort2(src0, src1, dst, tmp, src2, src3, exhausted=False)
        pl.store(out, dst, [0, 0])
```
