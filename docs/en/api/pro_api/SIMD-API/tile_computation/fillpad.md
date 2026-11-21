# pypto_pro.language.fillpad

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

Fills the padding region outside the valid data of a Tile, for padding tail-block data before subsequent computations. On the UB path, data is preserved according to the valid region of src, and the remaining specified region is filled. On the L1 Buffer path, only out is zero-filled, and the valid data must be loaded into out before the call. The fill value and layout constraints depend on the storage space in which the Tile resides. mode selects the address and shape relationship between the source and destination Tiles. For details about the mode definition, see [pypto_pro.language.FillPadMode](../basic_data_structures/FillPadMode.md).

## Prototype

```python
pypto_pro.language.fillpad(out: Tile, src: Tile, *, mode: FillPadMode=FillPadMode.NORMAL) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination tile, which is located in UB or L1 Buffer and stores the padding result. The 4-bit, 8-bit, 16-bit, and 32-bit storage types are supported. For details about the layout, padding, and mode requirements, see [Constraints](#constraints).|
| src | Input| Source Tile, which must reside in the same Buffer as out, with the same element bit width as out. On the UB path, the valid Shape of src specifies the data range to be preserved. On the L1 Buffer path, valid data is not copied from src. For details about the initialization requirements and zero-fill boundaries, see [Constraints](#constraints).|
| mode | Input| (Optional) [pypto_pro.language.FillPadMode](../basic_data_structures/FillPadMode.md) enumeration value. The default value is FillPadMode.NORMAL. It must be determined at compile time.|

## Return Value

None

## Constraints

### UB Tile

- out must have a valid pad attribute configured, for example, pypto_pro.language.TilePad.zero.
- The number of valid rows of src must not exceed the number of valid rows of out, and the number of valid rows of out must not exceed its physical row count. The number of valid columns of src must not exceed the physical column count of out.
- Within the valid rows of src, the fill range extends from after the valid columns of src to the end of the physical columns of out; after the valid rows of src, full rows are filled up to the valid row count of out. Physical rows of out after its valid rows are not guaranteed to be filled. To pad the entire destination Tile, set the valid Shape of out to its physical Shape.
- In NORMAL mode, the source and destination have the same physical Shape and different addresses.
- In EXPAND mode, the physical Shape of out must not be smaller than that of src in any dimension, and the source and destination have different addresses.
- In INPLACE mode, the source and destination have the same physical Shape and must share the same address. The valid data is retained at the original address.
- When 4-bit types are used, Shape counts logical elements, and the last dimension of the physical Shape must satisfy the packing requirement that two elements share one byte.

### L1 Buffer Tile

- When NORMAL mode and NZ layout are used, out and src must be of the same Tile type, including data type, physical Shape, valid Shape, and pad configuration.
- The current L1 Buffer path is used for zero-fill. Create the source and destination Tiles with pypto_pro.language.TilePad.null; other pad policies used for UB should not be applied directly to this path.
- Before the call, the valid data to be preserved must be loaded into out. This path fills only the padding of out and does not copy valid data from an independent src. The zero-fill boundary is determined by the valid Shape of out; the runtime valid Shape of src is not used.
- `pypto_pro.language.fillpad(tile, tile)` can be called on an initialized L1 Buffer Tile to zero-fill outside the existing valid data.
- EXPAND and INPLACE modes are intended for UB Tiles and are not used for L1 Buffer Tiles.

## Examples

### NORMAL Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src)
        pl.store(z, cur_dst, [0, 0])
```

### EXPAND Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_expand_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 16], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 16], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.EXPAND)
        pl.store(z, cur_dst, [0, 0])
```

### INPLACE Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_inplace_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec,
                           pad=pl.TilePad.zero, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0000, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.INPLACE)
        pl.store(z, cur_dst, [0, 0])
```
