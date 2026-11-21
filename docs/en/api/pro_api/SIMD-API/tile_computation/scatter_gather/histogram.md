# pypto_pro.language.histogram

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Generates cumulative counts for 256 bins per row based on the specified byte values of the source Tile elements. It can be used to determine bin boundaries in radix sort. The b-th bin in dst represents the number of elements that pass filtering and have a selected byte value less than or equal to b, not merely the frequency of elements equal to b.

For example, if the selected byte values are [0, 1, 1, 3], the first four bins are [1, 3, 3, 4].

## Prototype

```python
pypto_pro.language.histogram(dst: Tile, src: Tile, idx: Tile, *, is_msb: bool) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dst | Output| Destination UB Tile, with data type DT_UINT32 and pypto_pro.language.ND layout. The number of valid rows is the same as that of src; each row provides 256 cumulative count bins, with Shape [number of rows, 256].|
| src | Input| Source operand of type Tile; the elements to be counted. Its dtype is DT_UINT16 or DT_UINT32; the shape is arbitrary; the layout is pypto_pro.language.ND (row-major).|
| idx | Input| Index operand of type Tile; used for filtering when is_msb is False. Its dtype is DT_UINT8. When src is DT_UINT16, its shape has the same number of rows as src and 1 column, with pypto_pro.language.DN layout (column-major); when src is DT_UINT32 and is_msb is True, it is not used; when src is DT_UINT32 and is_msb is False, its shape has 3 rows and the same number of columns as src, with pypto_pro.language.ND layout (row-major).|
| is_msb | Input| Compile-time bool, which must be explicitly passed. True counts the most significant byte: bits 15–8 for DT_UINT16 and bits 31–24 for DT_UINT32, without high-byte filtering. False counts bits 7–0: for DT_UINT16, the high byte must match the corresponding row value in idx; for DT_UINT32, byte3, byte2, and byte1 must match the filter values in rows 0, 1, and 2 of idx, respectively.|

## Return Value

None

## Constraints

- src supports DT_UINT16 and DT_UINT32; dst is DT_UINT32; idx is DT_UINT8.
- src and dst use ND layout. For a DT_UINT16 source, idx uses DN layout, with each valid source row corresponding to one filter value; for a DT_UINT32 source, the low-byte mode uses a 3-row ND filter Tile.
- The output is a cumulative count that includes the current bin. When the frequency of an individual bin is needed, for the 0th bin in each row, directly take the 0th value of dst in that row; for the other bins, take the difference between adjacent cumulative values in that row.

## Examples

### Basic Usage

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

Result example:

<!-- pypto-doc-output:histogram:start -->
```bash
Input data src: [[0 257 514 771 1028 1285 1542 1799 ...], [32896 33153 33410 33667 33924 34181 34438 34695 ...], [256 513 770 1027 1284 1541 1798 2055 ...], [33152 33409 33666 33923 34180 34437 34694 34951 ...], ...]
Input data idx: [[0], [0], [0], [0], ...]
Output data out: [[1 2 3 4 5 6 7 8 ...], [0 0 0 0 0 0 0 0 ...], [0 1 2 3 4 5 6 7 ...], [1 1 1 1 1 1 1 1 ...], ...]
```
<!-- pypto-doc-output:histogram:end -->
