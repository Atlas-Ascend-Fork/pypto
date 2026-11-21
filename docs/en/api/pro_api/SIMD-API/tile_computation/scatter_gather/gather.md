# pypto_pro.language.gather

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

Supports two modes: index-based aggregation and index collection based on comparison conditions.

- **Index mode**: When cmp_mode is not explicitly passed, src is read according to the flat element indices in indices_or_k_value and written to out. This corresponds to [pypto_pro.language.scatter](scatter.md).
- **Comparison mode**: When cmp_mode is explicitly passed, comparison is performed against an integer threshold according to the rule corresponding to the type of src; the indices of elements that satisfy the condition are written to out, and the number of bytes written per row is written to the count Tile. This can be used for index filtering in scenarios such as TopK. For details about the type and value constraints, see [Constraints](#constraints).

## Prototype

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

Index mode uses the first three positional arguments, or additionally provides a fourth temporary Tile; cmp_mode is not passed. Comparison mode must provide five positional arguments and explicitly pass cmp_mode.

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination UB Tile. In index mode, its data type is the same as that of src, and its Shape and valid Shape must match the index result; in comparison mode, it uses DT_INT32 or DT_UINT32 and stores the selected indices, and its capacity must cover all elements that may be selected.|
| src | Input| Source UB Tile. In index mode, its flat elements are read; in comparison mode, it supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, and DT_FP32. For details about the comparison rules and mode constraints, see [Constraints](#constraints).|
| indices_or_k_value | Input| In index mode, this is an index Tile supporting DT_INT16, DT_UINT16, DT_INT32, and DT_UINT32, with indices in units of elements; 64-bit source data must use 32-bit indices. In comparison mode, this is an integer threshold Tile and must use DT_UINT16 or DT_UINT32; it does not need to have the same bit width as src, and floating-point Tiles are not used. Thresholds are stored contiguously in the order of valid rows of the source, with the i-th element corresponding to the i-th row of src.|
| tmp_or_cdst | Output| Optional; default is None. In index mode, this is a compatibility temporary Tile and may be omitted; in comparison mode, an output count Tile must be passed in, storing the number of bytes written to out for each row. DT_INT32 or DT_UINT32 is supported. For 32-bit indices, the number of valid elements equals this byte count divided by 4.|
| tmp | Input| Optional; default is None. This parameter is not passed in index mode; in comparison mode, a compatibility temporary Tile must be provided as the fifth positional argument.|
| cmp_mode | Input| Comparison mode control parameter, a compile-time integer. Explicitly passing 0 indicates equality comparison, and 4 indicates greater-than comparison; currently only these two values are supported. This parameter must be omitted in index mode; explicitly passing 0 does not mean disabling comparison.|
| offset | Input| Optional, a compile-time integer (default value: 0). Used only in comparison mode; it is added to the collected index values, does not change the read position in src, and does not apply an offset to the input indices in index mode.|

## Return Value

None

## Constraints

- All Tiles must be located in UB and allocated storage space according to the layout requirements of the corresponding Tile; Tiles used for different purposes should use non-overlapping regions.
- In index mode, out and src must have the same data type, and the valid region of the index Tile must cover the valid region of out. Each input index must point to a readable element in src.
- In comparison mode, src, the threshold, out, and the count Tile use a row-major layout. The count Tile and the fifth temporary Tile must be provided; the four-argument call of index mode must not be combined with cmp_mode.
- In comparison mode, only the first "counted byte count/4" indices in each row of out are valid. The count Tile records the number of bytes, not the number of elements.

### Type and Value Constraints in Comparison Mode

- The comparison mode is used for filtering non-negative values against an integer threshold. When the source is a signed integer type, negative values should not be interpreted according to the usual signed-number magnitude relationship.
- When src is DT_UINT16, only equality filtering is currently used, that is, cmp_mode=0; the current type dispatch does not provide a separate greater-than comparison path for this type, so cmp_mode=4 cannot be used to obtain a greater-than filtering result.
- When src is DT_FP32, it is first converted to DT_INT32 according to the underlying ROUND_R rule, and the conversion result is then interpreted as DT_UINT32 and compared with the integer threshold. Therefore, this is not equivalent to directly performing equality or magnitude comparison on the original floating-point value. For example, when a src element is 20.25 and the threshold is 20, the converted value is 20, and it will be selected under cmp_mode=0.
- A DT_FP32 source must contain finite values, and the converted values must fall within the non-negative range of DT_INT32; NaN, infinity, or out-of-range inputs are not applicable to the above filtering semantics.
- When src is DT_FP16, the integer threshold is converted through a 16-bit signed integer conversion and then compared with the source data promoted to DT_FP32; the threshold should be an integer in the range [0, 32,767], and a DT_FP16 bit pattern should not be passed in place of the integer threshold.

## Examples

### Index Mode

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

Result example:

<!-- pypto-doc-output:gather:start -->
```bash
Input src: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 67 67.25 67.5 67.75 68 68.25 68.5 68.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 99 99.25 99.5 99.75 100 100.25 100.5 100.75 ...], ...]
Input indices: [[127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112 ...], [255 254 253 252 251 250 249 248 247 246 245 244 243 242 241 240 ...], [383 382 381 380 379 378 377 376 375 374 373 372 371 370 369 368 ...], [511 510 509 508 507 506 505 504 503 502 501 500 499 498 497 496 ...], ...]
Output data dst: [[32.75 32.5 32.25 32 31.75 31.5 31.25 31 30.75 30.5 30.25 30 29.75 29.5 29.25 29 ...], [64.75 64.5 64.25 64 63.75 63.5 63.25 63 62.75 62.5 62.25 62 61.75 61.5 61.25 61 ...], [96.75 96.5 96.25 96 95.75 95.5 95.25 95 94.75 94.5 94.25 94 93.75 93.5 93.25 93 ...], [128.75 128.5 128.25 128 127.75 127.5 127.25 127 126.75 126.5 126.25 126 125.75 125.5 125.25 125 ...], ...]
```
<!-- pypto-doc-output:gather:end -->

### Comparison Mode

The Kernel below collects the positions of elements greater than 20 and adds 16 to the output indices. In indices_out, only the first byte_count_out[0, 0]/4 indices are valid; the remaining positions should not be used as aggregation results.

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
        # Collect the indices of elements greater than 20; add the offset to the collected indices.
        pl.gather(indices, tsrc, key, count, tmp, cmp_mode=4, offset=16)
        pl.store(indices_out, indices, [0, 0])
        pl.store(byte_count_out, count, [0, 0])
```

If the first 8 elements of src are [0, 10, 20, 30, 20, 10, 40, 20] and all remaining elements are 0, the valid output indices are [19, 22], and the counted byte count is 8.
