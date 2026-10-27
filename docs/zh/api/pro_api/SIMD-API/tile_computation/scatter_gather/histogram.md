# pypto_pro.language.histogram

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

按源Tile元素的指定字节值生成每行256个桶的累计计数，可用于基数排序中确定桶边界。dst中第b个桶表示：通过过滤且选中字节值小于或等于b的元素个数，而不是仅等于b的频数。

例如选中的字节值为[0, 1, 1, 3]，则前4个桶的结果为[1, 3, 3, 4]。

## 函数原型

```python
pypto_pro.language.histogram(dst: Tile, src: Tile, idx: Tile, *, is_msb: bool) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 目的UB Tile，数据类型为DT_UINT32，采用pypto_pro.language.ND布局。有效行数与src一致，每行提供256个累计计数桶，Shape为[行数, 256]。 |
| src | 输入 | 源操作数，Tile类型，待统计的元素。dtype为DT_UINT16或DT_UINT32；shape任意；布局为pypto_pro.language.ND（行主序）。 |
| idx | 输入 | 索引操作数，Tile类型，is_msb为False时用于过滤。dtype为DT_UINT8：src为DT_UINT16时，shape行数与src一致、列数为1，布局为pypto_pro.language.DN（列主序）；src为DT_UINT32且is_msb为True时不使用；src为DT_UINT32且is_msb为False时，shape行数为3、列数与src一致，布局为pypto_pro.language.ND（行主序）。 |
| is_msb | 输入 | 编译期bool，必须显式传入。True统计最高字节：DT_UINT16为bits 15～8，DT_UINT32为bits 31～24，不执行高位过滤。False统计bits 7～0：DT_UINT16要求高字节与idx对应行值匹配；DT_UINT32要求byte3、byte2、byte1分别与idx的第0、1、2行过滤值匹配。 |

## 返回值说明

无。

## 约束说明

- src支持DT_UINT16、DT_UINT32；dst为DT_UINT32；idx为DT_UINT8。
- src和dst采用ND布局。DT_UINT16源的idx采用DN布局，每个源有效行对应一个过滤值；DT_UINT32源的低字节模式使用3行ND过滤Tile。
- 输出为包含当前桶的累计计数。需要单个桶的频数时，每行第0桶直接取该行dst的第0个值，其余桶取该行相邻累计值之差。

## 调用示例

### 基本用法

```python
import pypto_pro.language as pl

ROWS = 32
COLS = 128
IDX_COLS_DN = 1


@pl.jit(auto_mutex=True)
def histogram_kernel(
    src: pl.Tensor[[ROWS, COLS], pl.DT_UINT16],
    idx: pl.Tensor[[ROWS, IDX_COLS_DN], pl.DT_UINT8],
    out: pl.Tensor[[ROWS, 256], pl.DT_UINT32],
):
    pl.system.bar_all()
    tt_src = pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_UINT16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    tt_idx = pl.TileType(shape=[ROWS, IDX_COLS_DN], dtype=pl.DT_UINT8,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tt_dst = pl.TileType(shape=[ROWS, 256], dtype=pl.DT_UINT32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x2000, mutex_ids=[1])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x2020, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, idx, [0, 0])
        pl.histogram(cur_dst, cur_src, cur_idx, is_msb=True)
        pl.store(out, cur_dst, [0, 0])
```

结果示例如下：

<!-- pypto-doc-output:histogram:start -->
```bash
输入数据src：[[0 257 514 771 1028 1285 1542 1799 ...], [32896 33153 33410 33667 33924 34181 34438 34695 ...], [256 513 770 1027 1284 1541 1798 2055 ...], [33152 33409 33666 33923 34180 34437 34694 34951 ...], ...]
输入数据idx：[[0], [0], [0], [0], ...]
输出数据out：[[1 2 3 4 5 6 7 8 ...], [0 0 0 0 0 0 0 0 ...], [0 1 2 3 4 5 6 7 ...], [1 1 1 1 1 1 1 1 ...], ...]
```
<!-- pypto-doc-output:histogram:end -->
