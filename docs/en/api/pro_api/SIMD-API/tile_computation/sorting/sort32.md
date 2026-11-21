# pypto_pro.language.sort32

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

Grouped sort: This API groups every 32 consecutive valid elements of each row in src, sorts each group in descending order by value, and reorders idx using the same permutation. Each input element generates one fixed 8-byte value-index record, which is written to dst.

The byte layout of the record is as follows.

| src dtype | 8-Byte Record Layout| Number of dst Elements/Record|
|---|---|---|
| DT_FP32 | [value_f32(4Byte), index_u32(4Byte)] | 2 |
| DT_FP16 | [value_f16(2Byte), padding(2Byte, 0), index_u32(4Byte)] | 4 |

For the tail block scenario where fewer than 32 elements are available, the tmp parameter needs to be provided as an intermediate buffer.

## Prototype

```python
pypto_pro.language.sort32(dst: Tile, src: Tile, idx: Tile, tmp: Optional[Tile]=None) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dst | Output| UB, RowMajor Tile; its dtype is the same as that of src and can only be DT_FP16 or DT_FP32. If src.valid_shape=[R,C], then dst.valid_shape must be [R,4C] for FP16 or [R,2C] for FP32. Since the hardware writes records in complete 32-element groups, the physical column count of dst must be no less than 4 * ceil32(C) or 2 * ceil32(C), respectively.|
| src | Input| UB, RowMajor Tile, with dtype DT_FP16 or DT_FP32; each row is grouped by 32 consecutive valid values, and each group is sorted independently.|
| idx | Input| UB, RowMajor, DT_UINT32 Tile; its valid column count must be at least C, and its physical column count must be at least ceil32(C). In tail blocks, index values beyond C do not need to be initialized. When idx.valid_shape[0]=1, the same row of indices is broadcast to all data rows of src; otherwise, the valid row count must equal R and indices are used row by row.|
| tmp | Output| Optional, UB, RowMajor Tile; its dtype must be the same as that of src. It is required only when C % 32 != 0; see the formula below for the minimum column count. Its physical address must not overlap with src, idx, or dst.|

### tmp size

Only non-32-byte-aligned tail blocks require tmp. Let C=src.valid_shape[1], B be the element byte size (2 for FP16, 4 for FP32), and ceil32(C) denote rounding C up to a multiple of 32. The minimum required tmp column count is:

$$
tmpCols_{min}=\begin{cases}
\operatorname{ceil32}(C), & C\times B\le 8160 \\
32, & C\times B>8160
\end{cases}
$$

The first path copies the entire row into tmp and then pads the tail block; the second path copies only the last tail block of fewer than 32 elements. The padding value is negative infinity, so that padding elements are sorted to the end of the descending result. When C % 32 = 0, the three-argument form without tmp should be used.

## Return Value

None. The sorted value-index records are written to dst.

## Constraints

1. The buffers for src, idx, dst, and the optional tmp must not overlap. The base address of each Tile must at least satisfy the 32-byte alignment requirement of UB Tiles. In addition, the src address of each hardware iteration must be aligned to 128 bytes (FP32) or 64 bytes (FP16); the idx address must be aligned to 128 bytes; and the dst address must be aligned to 256 bytes. When grouping by consecutive 32 elements, the addresses of adjacent iterations naturally advance by the above step sizes, so the start addresses of all Tiles should be planned according to these requirements.
2. Each hardware sort repeat processes 32 values, and a single hardware call supports up to 255 repeats; longer rows are split by the underlying implementation into multiple calls, but ordering is still guaranteed only within each 32-element group.
3. Results are sorted in descending order by value; when values are equal, the record with the smaller index comes first.
4. The relative order of NaN is determined by the underlying floating-point comparison behavior, and its fixed position in the result should not be relied upon.
5. The API processes only the data within src.valid_shape; content outside the valid region in the physical shape does not participate in sorting.

## Examples

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

### Tail Block Scenario

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
