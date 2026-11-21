# pypto_pro.language.gatherb

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

Reads 32-byte data blocks (for example, 16 DT_FP16 elements per block) from the source Tile at the byte offsets specified by offsets, and packs them into the destination Tile.

## Prototype

```python
pypto_pro.language.gatherb(out: Tile, src: Tile, offsets: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of type Tile, which stores the aggregation result obtained using byte offsets. The dtype must be the same as that of src. The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, and DT_FP32. The shape and valid_shape must match the expected output.|
| src | Input| Source operand of type Tile. The dtype and shape must be the same as those of out.|
| offsets | Input| Byte offset tile. The supported data types are DT_INT32 and DT_UINT32. Each valid offset specifies the starting position of a 32-byte data block; the offset must be non-negative, and the block must lie entirely within the readable storage range of src. Let C be the number of valid elements per output row and B be the element byte size; then at least ceil(C * B/32) valid offsets are required per row. The physical shape must also satisfy the layout alignment requirements of the offset Tile itself.|

## Return Value

None

## Constraints

- A complete 32-byte block must satisfy offset + 32 not exceeding the readable byte range of src; merely checking that offset is less than the total source byte count is not sufficient to avoid out-of-bounds access.
- out, src, and offsets must be located in UB, and the output must have the same element bit width as the source. The caller must provide all blocks according to the actual read range, and the tail block must also have valid source storage space.

## Examples

### Basic Usage

```python
import pypto_pro.language as pl

DTYPE_SIZE = 2          # FP16
BLOCK_BYTES = 32
BLOCK_ELEMS = BLOCK_BYTES // DTYPE_SIZE   # 16
ROWS, COLS = 64, 128
OFFSETS_PER_ROW = COLS // BLOCK_ELEMS     # 8


@pl.jit(auto_mutex=True)
def gatherb_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    offsets: pl.Tensor[[64, 8], pl.DT_UINT32],
    dst: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_offsets_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 8], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4800, mutex_ids=[2])
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_offsets = tile_offsets_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.load(tile_offsets, offsets, [0, 0])
        pl.gatherb(tile_dst, tile_src, tile_offsets)
        pl.store(dst, tile_dst, [0, 0])
```

Result example:

<!-- pypto-doc-output:gatherb:start -->
```bash
Input data src: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
Input data offsets: [[0 32 64 96 128 160 192 224], [256 288 320 352 384 416 448 480], [512 544 576 608 640 672 704 736], [768 800 832 864 896 928 960 992], ...]
Output data dst: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
```
<!-- pypto-doc-output:gatherb:end -->
