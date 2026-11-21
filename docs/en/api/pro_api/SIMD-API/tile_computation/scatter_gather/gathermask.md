# pypto_pro.language.gathermask

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

Extracts columns from the source Tile into the destination Tile in bit-pattern mode; the columns to be extracted are determined by the pattern_mode parameter.

## Prototype

```python
pypto_pro.language.gathermask(out: Tile, src: Tile, *, pattern_mode: int) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination UB Tile, with the same data type as src, storing the elements extracted according to the pattern. The capacity must be sufficient to cover all selected elements. For details about valid shape settings, see [Constraints](#constraints).|
| src | Input| Source operand of the Tile type, which is located in the UB, in row-major order. The dtype supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_INT64, DT_UINT64, DT_FP8E4M3FN, DT_FP8E5M2, and DT_HF8. The data types of out must be the same as those of src.|
| pattern_mode | Input| Bit pattern, a compile-time constant integer ranging from 1 to 7, determining which columns are extracted:<br>1: even columns, that is, src[:, 0::2]<br>2: odd columns, that is, src[:, 1::2]<br>3: the first column in every four columns, that is, src[:, 0::4]<br>4: the second column in every four columns, that is, src[:, 1::4]<br>5: the third column in every four columns, that is, src[:, 2::4]<br>6: the fourth column in every four columns, that is, src[:, 3::4]<br>7: all columns, equivalent to copy|

## Return Value

None

## Constraints

- pattern_mode must be determined at compile time and takes a value from 1 to 7.
- out has the same data type as src, uses a row-major layout, and must satisfy the alignment requirements of the corresponding Tile.
- For 8-, 16-, and 32-bit elements, the number of columns extracted per row is determined by pattern_mode, and out must provide contiguous storage space for that result.
- For 64-bit elements, the extracted results of all valid source rows are written out by concatenation, without inserting inter-row padding according to the physical column count of out. Let K be the number of elements selected per row and R be the number of valid rows; contiguous space for R * K elements must be provided, for example, by using a destination Tile with valid shape [1, R * K].

## Examples

### Basic Usage

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p1_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.gathermask(tile_dst, tile_src, pattern_mode=1)
        pl.store(dst, tile_dst, [0, 0])
```

Result example:

<!-- pypto-doc-output:gathermask:start -->
```bash
Input data src: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
Input data pattern_mode: 1
Output data dst: [[1 1.5 2 2.5 3 3.5 4 4.5 ...], [33 33.5 34 34.5 35 35.5 36 36.5 ...], [65 65.5 66 66.5 67 67.5 68 68.5 ...], [97 97.5 98 98.5 99 99.5 100 100.5 ...], ...]
```
<!-- pypto-doc-output:gathermask:end -->

### Extracting Odd Columns

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p2_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    source_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    output_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    gsrc = pl.make_tile_group(type=source_type, addrs=0x0000, mutex_ids=[0])
    gout = pl.make_tile_group(type=output_type, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tsrc = gsrc.current()
        tout = gout.current()
        pl.load(tsrc, src, [0, 0])
        pl.gathermask(tout, tsrc, pattern_mode=2)
        pl.store(dst, tout, [0, 0])
```

### Copying All Columns

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p7_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    source_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    output_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    gsrc = pl.make_tile_group(type=source_type, addrs=0x0000, mutex_ids=[0])
    gout = pl.make_tile_group(type=output_type, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tsrc = gsrc.current()
        tout = gout.current()
        pl.load(tsrc, src, [0, 0])
        pl.gathermask(tout, tsrc, pattern_mode=7)
        pl.store(dst, tout, [0, 0])
```
