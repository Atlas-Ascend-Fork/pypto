# pypto_pro.language.sort32

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

分组排序：把src每行按连续32个有效元素分组并按值降序排序，同时按相同置换重排idx。每个输入元素生成一条固定8字节的value-index记录并写入dst。

记录的字节布局如下：

| src dtype | 8字节记录布局 | 每条记录占dst元素数 |
|---|---|---|
| DT_FP32 | [value_f32(4Byte), index_u32(4Byte)] | 2 |
| DT_FP16 | [value_f16(2Byte), padding(2Byte, 0), index_u32(4Byte)] | 4 |

对于不足32元素的尾块场景，需提供tmp参数作为中间缓冲。

## 函数原型

```python
pypto_pro.language.sort32(dst: Tile, src: Tile, idx: Tile, tmp: Optional[Tile]=None) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | UB、RowMajor Tile，dtype与src相同且只能为DT_FP16或DT_FP32。若src.valid_shape=[R,C]，则dst.valid_shape须为FP16时的[R,4C]或FP32时的[R,2C]；硬件按完整32元素组写出记录，因此dst物理列数须分别不少于4 * ceil32(C)或2 * ceil32(C)。 |
| src | 输入 | UB、RowMajor Tile，dtype为DT_FP16或DT_FP32；每行按连续32个有效值分组，组与组之间独立排序。 |
| idx | 输入 | UB、RowMajor、DT_UINT32 Tile；有效列数须至少为C，物理列数须至少为ceil32(C)，尾块中超出C的索引值无需初始化。idx.valid_shape[0]=1时同一行索引广播给src所有数据行；否则有效行数须等于R并逐行使用。 |
| tmp | 输出 | 可选，UB、RowMajor Tile，dtype须与src一致；仅当C % 32 != 0时必传，最小列数见下方公式。物理地址不得与src、idx或dst重叠。 |

### tmp尺寸

仅非32对齐尾块需要tmp。设C=src.valid_shape[1]，B为元素字节数（FP16为2，FP32为4），ceil32(C)表示向上取整到32的倍数。所需的最小tmp列数为：

$$
tmpCols_{min}=\begin{cases}
\operatorname{ceil32}(C), & C\times B\le 8160 \\
32, & C\times B>8160
\end{cases}
$$

前一条路径把整行复制到tmp后填充尾块，后一条路径只复制最后一个不足32元素的尾块。填充值为负无穷，使填充元素排在降序结果末尾。C % 32 = 0时应使用不带tmp的三参数形式。

## 返回值说明

无返回值。排序后的value-index记录写入dst。

## 约束说明

1. src、idx、dst以及可选tmp的缓冲区不得重叠。Tile首地址至少须满足UB Tile的32字节对齐要求；此外，每次硬件迭代的src地址须按128字节（FP32）或64字节（FP16）对齐，idx地址须按128字节对齐，dst地址须按256字节对齐。按连续32元素分组时，相邻迭代地址会自然按上述步长递增，因此应按该要求规划各Tile的起始地址。
2. 每个硬件排序repeat处理32个值，单次硬件调用的repeat上限为255；更长的行由底层拆分为多次调用，但仍只保证每个32元素分组内部有序。
3. 结果按value降序排列；value相同时，index较小的记录排在前面。
4. NaN的相对顺序由底层浮点比较行为决定，不应依赖其在结果中的固定位置。
5. 接口只处理src.valid_shape内的数据，物理shape中超出有效区域的内容不参与排序。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sort32_kernel(
    a: pl.Tensor[[1, 32], pl.DT_FP16],
    idx_in: pl.Tensor[[1, 32], pl.DT_UINT32],
    sorted_out: pl.Tensor[[1, 128], pl.DT_FP16],
):
    tt_src = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_idx = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x0080, mutex_ids=[1])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x0100, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        cur_idx = tile_idx.current()
        pl.load(cur_src, a, [0, 0])
        pl.load(cur_idx, idx_in, [0, 0])
        pl.sort32(cur_dst, cur_src, cur_idx)
        pl.store(sorted_out, cur_dst, [0, 0])
```

### 尾块场景

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sort32_tail_kernel(
    a: pl.Tensor[[1, 16], pl.DT_FP16],
    idx_in: pl.Tensor[[1, 16], pl.DT_UINT32],
    sorted_out: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tt_src = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[1, 128], valid_shape=[1, 64], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec)
    tt_idx = pl.TileType(shape=[1, 32], valid_shape=[1, 16], dtype=pl.DT_UINT32,
                         target_memory=pl.MemorySpace.Vec)
    tt_tmp = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x0080, mutex_ids=[1])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x0100, mutex_ids=[2])
    tile_tmp = pl.make_tile_group(type=tt_tmp, addrs=0x0200, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        cur_idx = tile_idx.current()
        cur_tmp = tile_tmp.current()
        pl.load(cur_src, a, [0, 0])
        pl.load(cur_idx, idx_in, [0, 0])
        pl.sort32(cur_dst, cur_src, cur_idx, cur_tmp)
        pl.store(sorted_out, cur_dst, [0, 0])
```
