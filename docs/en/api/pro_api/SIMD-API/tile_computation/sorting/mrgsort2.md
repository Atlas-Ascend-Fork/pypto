# pypto_pro.language.mrgsort2

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

Multi-way merge sort: This API merges 2 to 4 source lists that are already sorted in descending order by value into a single descending output. Each source Tile holds fixed 8-byte value-index records, and the record layout is the same as the [sort32](sort32.md) output: for FP32, [value_f32(4Byte), index_u32(4Byte)]; for FP16, [value_f16(2Byte), padding(2Byte, 0), index_u32(4Byte)].

The underlying vmrgsort4 first writes the merge result into tmp, and then copies the prefix specified by dst.valid_shape into dst. Therefore, dst may be smaller than the sum of all inputs, but tmp must still accommodate the full total capacity of the input lists.

exhausted controls whether the hardware pauses merging when one input list is exhausted. It is used to handle source lists of unequal lengths in multi-step merging; it is a control bit for the entire call and is not used to specify a particular source. The underlying implementation records the number of elements actually consumed from each list, but the current Pro interface does not return these counts.

## Prototype

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

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| src0 | Input| UB, RowMajor, single-row Tile; its dtype is DT_FP16 or DT_FP32. Its contents must be 8-byte value-index records sorted in descending order by value.|
| src1 | Input| UB, RowMajor, single-row Tile; its dtype is DT_FP16 or DT_FP32. Its contents must be 8-byte value-index records sorted in descending order by value.|
| dst | Output| UB, RowMajor, single-row Tile; its dtype is the same as that of all sources. The number of output columns is controlled by dst.valid_shape[1] and may be smaller than the total column count of all sources, but must not exceed the amount of data that can be merged.|
| tmp | Output| UB, RowMajor, single-row Tile, with the same dtype as all sources; its physical column count must be at least the sum of the physical column counts of all sources; see the formula below.|
| *args | Input| Optional additional sorted source Tiles (supporting 3-way or 4-way merge), making the total number of sources 2 to 4; each Tile must satisfy the same type, layout, and ordering requirements as src0/src1.|
| exhausted | Input| Optional compile-time bool, default False; True means that the current hardware merge stops when any input list is exhausted. The current Pro interface does not return the actual consumed count of each list.|

### tmp size

Let the physical column counts of the source Tiles participating in the merge be src0.shape[1], src1.shape[1], etc. Then:

$$
tmp.shape[1]\ge\sum_k src_k.shape[1]
$$

That is:

- 2-way: tmp.shape[1] >= src0.shape[1] + src1.shape[1];
- 3-way: add src2.shape[1] to the above;
- 4-way: add src2.shape[1] + src3.shape[1] to the above.

tmp is counted in Tile storage elements and must not be calculated in terms of the number of value-index records. Each FP32 record occupies 2 columns, and each FP16 record occupies 4 columns.

## Return Value

None. The merge result is written to dst. The actual number of consumed elements in each input list is not returned.

## Constraints

1. All sources, dst, and tmp must be UB, RowMajor, and single-row Tiles with exactly the same dtype, and must use non-overlapping UB regions aligned to 32 bytes.
2. The valid column count of each source Tile must be an integer multiple of a complete 8-byte record: a multiple of 2 for FP32 and a multiple of 4 for FP16. The number of 8-byte records contained in each source list must not exceed 4095, that is, the valid column count must not exceed 8190 for FP32 or 16380 for FP16.
3. Each source list must already be sorted in descending order by value before the call. The API does not check input ordering; if the input is unordered, the output is not guaranteed to be ordered.
4. The address ranges of all Tiles must satisfy the UB capacity check of the target platform. SIMD Tile addresses are checked against the physical capacity of the target platform; mixed Kernels containing SIMT are also subject to static and dynamic user UB quotas, and a uniform 248 KiB limit cannot be used in place of these two conditions. The capacity of the temporary Tile must still satisfy the tmp size requirements on this page.
5. When exhausted=True, the Pro API does not expose the underlying MrgSortExecutedNumList to the caller. If the algorithm needs to know the exact consumed count of each source, the current API cannot provide this information.
6. Results are sorted in descending order by value; when values are equal, the record with the smaller index comes first. The relative order of NaN is determined by the underlying floating-point comparison behavior.

## Examples

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

### Three-Way Merge

Each input must be a value-index record sorted in descending order.

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

### Four-Way Merge

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
