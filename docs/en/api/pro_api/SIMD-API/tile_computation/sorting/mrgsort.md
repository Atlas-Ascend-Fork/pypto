# pypto_pro.language.mrgsort

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

Four-way block merge: This API merges each group of 4 consecutive, equal-length blocks in src—each already sorted in descending order by value—into a single descending block and writes it to dst. This API operates on fixed 8-byte value-index records; it does not perform a full sorting on arbitrary unsorted data.

The record layout is the same as the [sort32](sort32.md) output: an FP32 record is [value_f32(4Byte), index_u32(4Byte)], and an FP16 record is [value_f16(2Byte), padding(2Byte, 0), index_u32(4Byte)].

Typical scenario: a preprocessing step for TopK sorting—sort within each block first, then use [pypto_pro.language.mrgsort2](mrgsort2.md) for multi-way merging.

## Prototype

```python
pypto_pro.language.mrgsort(dst: Tile, src: Tile, *, block_len: int) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dst | Output| UB, RowMajor, single-row Tile; its dtype, physical shape, and valid_shape must be consistent with those of src.|
| src | Input| UB, RowMajor, single-row Tile, with dtype DT_FP16 or DT_FP32; its contents must be 8-byte value-index records, and each block to be merged is already sorted in descending order by value.|
| block_len | Input| Compile-time positive integer, measured in Tile storage elements rather than in the number of value-index records. Each value-index record occupies a fixed 8 bytes, so block_len * sizeof(dtype) must be a multiple of 8: in the FP32 case, block_len must be a multiple of 2; in the FP16 case, it must be a multiple of 4. The number of records per block, block_len * sizeof(dtype)/8, must be in the range [1, 4095]. For example, block_len=64 means 32 FP32 records or 16 FP16 records per block.|

## Return Value

None. The merge result is written to dst.

## Constraints

1. src.valid_shape[1] must be an integer multiple of 4 * block_len. Every 4 consecutive blocks form one four-way merge group, and multiple groups are independent of one another.
2. repeatTimes = src.valid_shape[1]/(4 * block_len) must be in the range [1, 255].
3. Each input block in src must be pre-sorted in descending order by value. Typically, [sort32](sort32.md) is first used to generate sorted records, and then mrgsort is called with increasing block_len.
4. src and dst must use non-overlapping UB regions aligned to 32 bytes; this API does not guarantee in-place merge results.
5. Results are sorted in descending order by value; when values are equal, the record with the smaller index comes first. The relative order of NaN is determined by the underlying floating-point comparison behavior.
6. block_len and the related shape constraints are measured in Tile storage elements and must not be mistaken for the number of value-index records; the converted number of value-index records per block must not exceed 4095.

## Examples

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
