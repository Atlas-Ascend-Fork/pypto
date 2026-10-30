# pypto_pro.language.mrgsort

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

四路块归并：将src中的每组4个连续、等长且已按值降序排列的块归并为一个降序块，并写入dst。本接口操作固定8字节的value-index记录，不是对任意未排序数据执行完整排序。

记录布局与[sort32](sort32.md)输出一致：FP32记录为[value_f32(4Byte), index_u32(4Byte)]，FP16记录为[value_f16(2Byte), padding(2Byte, 0), index_u32(4Byte)]。

典型场景：TopK排序的预处理步骤，先对每个块内部排序，再用[pypto_pro.language.mrgsort2](mrgsort2.md)多路归并。

## 函数原型

```python
pypto_pro.language.mrgsort(dst: Tile, src: Tile, *, block_len: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | UB、RowMajor、单行Tile；dtype、物理shape和valid_shape须与src一致。 |
| src | 输入 | UB、RowMajor、单行Tile，dtype为DT_FP16或DT_FP32；内容须为8字节value-index记录，每个待归并块内部已按value降序排列。 |
| block_len | 输入 | 编译期正整数，单位为Tile存储元素而不是value-index记录数。每条value-index记录固定占8字节，因此block_len * sizeof(dtype)必须是8的倍数：FP32场景block_len须为2的倍数，FP16场景须为4的倍数。每块记录数block_len * sizeof(dtype) / 8须位于[1, 4095]。例如block_len=64分别表示每块32条FP32记录或16条FP16记录。 |

## 返回值说明

无返回值。归并结果写入dst。

## 约束说明

1. src.valid_shape[1]必须是4 * block_len的整数倍。每连续4个块构成一组四路归并，多个组彼此独立。
2. repeatTimes = src.valid_shape[1] / (4 * block_len)必须位于[1,255]。
3. src中每个输入块必须预先按value降序排列。通常先用[sort32](sort32.md)生成有序记录，再逐级增大block_len调用mrgsort。
4. src和dst须使用互不重叠且按32字节对齐的UB区域；本接口不保证原地归并结果。
5. 结果按value降序排列；value相同时，index较小的记录排在前面。NaN的相对顺序由底层浮点比较行为决定。
6. block_len及相关shape约束按Tile存储元素计算，不得误用value-index记录数；换算后的每块value-index记录数不得超过4095。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mrgsort_kernel(
    a: pl.Tensor[[1, 1024], pl.DT_FP16],
    sorted_out: pl.Tensor[[1, 1024], pl.DT_FP16],
):
    tt = pl.TileType(shape=[1, 1024], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_dst = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[1])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, a, [0, 0])
        pl.mrgsort(cur_dst, cur_src, block_len=256)
        pl.store(sorted_out, cur_dst, [0, 0])
```
