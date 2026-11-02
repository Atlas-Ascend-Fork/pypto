# Tile Creation and Operations

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-27T02:08:14.124Z pushedAt=2026-10-27T10:37:55.653Z nl=2 -->

A Tile represents a two-dimensional data block in the on-chip Buffer of the AI Core. It records information such as its shape, data type, memory space, layout, and bound address, and serves as the basic object of operation for data movement and computation APIs.

## Tile Creation Process

Creating a Tile typically involves the following steps:

1. Use TileType to describe the Tile specification.
2. Use make_tile to create a single Tile, or use make_tile_group to create a group of rotatable Tiles.
3. Use next(), current(), previous(), or subscript access on the TileGroup to obtain a specific Tile.
4. In the tail block scenario, set valid_shape based on the actual valid range, and use reinterpret to create a new view sharing the same buffer when needed.

Tile and Tensor serve different purposes: Tensor describes a data view in GM, while Tile represents a data block in the AI Core on-chip Buffer. For Tensor declaration and view operations, see [Tensor Creation and Operations](tensor_creation_and_operations.md).

## Describing a Tile with TileType

pypto_pro.language.TileType only describes the Tile specification and does not allocate on-chip space. Its main parameters are as follows:

```python
tile_type = pypto_pro.language.TileType(
    shape=[64, 128],
    dtype=pypto_pro.language.DT_FP16,
    target_memory=pypto_pro.language.MemorySpace.Vec,
    valid_shape=None,
    layout=None,
    fractal=None,
    pad=None,
    compact=None,
)
```

| Parameter | Description |
|:---|:---|
| shape | Physical shape of the Tile. Currently, only a two-dimensional positive integer shape is supported. |
| dtype | Data type of the Tile elements. |
| target_memory | On-chip memory space where the Tile resides. The default value is MemorySpace.Vec. |
| valid_shape | Valid region of the Tile. If omitted, the dynamic mode is used by default, which is equivalent to [-1, -1]. The tail block is set at runtime through set_validshape. |
| layout | Physical layout of the Tile, such as ND, DN, NZ, ZN, NN, or ZZ. If omitted, the default value is derived based on the memory space. |
| fractal | Fractal size. The default value can be derived for some memory spaces and data types. |
| pad | Padding mode of the invalid region. |
| compact | Compact arrangement of the valid Tile data in the buffer. |

valid_shape, pad, and compact are mainly used for tail blocks and special layout scenarios. They do not change the buffer size bound by make_tile or make_tile_group. For complete constraints, see [pypto_pro.language.TileType](../../../../api/pro_api/SIMD-API/basic_data_structures/TileType.md).

### Selecting the On-Chip Memory Space

target_memory determines the physical buffer to which the Tile is bound and the data paths that can be used:

| pypto_pro.language.MemorySpace | Physical Buffer | Typical Use Cases |
|:---|:---|:---|
| Vec | UB | Input, output, and intermediate results of vector data |
| Mat | L1 Buffer | Cube staging between GM and L0A Buffer/L0B Buffer |
| Left | L0A Buffer | Left operand of cube multiplication |
| Right | L0B Buffer | Right operand of cube multiplication |
| Acc | L0C Buffer | Accumulation result of cube multiplication |
| Bias | BiasTable Buffer | Fused bias of cube multiplication |
| Scaling | Fixpipe Buffer | Per-channel quantization/dequantization parameters of the FIX data path |
| ScaleLeft | L0A_MX Buffer | Left quantization coefficient cube of MX cube multiplication |
| ScaleRight | L0B_MX Buffer | Right quantization coefficient cube of MX cube multiplication |

After selecting the memory space, you must also comply with the constraints of the corresponding buffer on layout, fractal, dtype, capacity, and address alignment. For detailed enumeration descriptions, see [pypto_pro.language.MemorySpace](../../../../api/pro_api/SIMD-API/basic_data_structures/MemorySpace.md).

## Choosing a Tile Creation Method

After determining the TileType, you can use make_tile to create a Tile bound to a fixed address, where the developer manages the access timing and synchronization; or use make_tile_group to create a group of Tiles with the same specification, uniformly managing addresses, rotation, and optional automatic synchronization metadata. Choose based on the buffering method and synchronization requirements:

| Requirement | Recommended Method |
|:---|:---|
| Create a Tile at a fixed address and manually control synchronization | make_tile |
| Create a Tile used only temporarily in local computation, without rotation or automatic synchronization | make_tile |
| Create a regular single buffer with automatic synchronization | make_tile_group, configure one mutex ID |
| Create double buffering or N-buffering | make_tile_group, configure multiple mutex IDs based on the buffer depth |
| Create a rotating Tile but manage synchronization manually | make_tile_group, use depth without configuring a mutex ID |

## Creating a Single Tile with make_tile

pypto_pro.language.make_tile binds a fixed address range in the memory space specified by TileType:

```python
tile = pypto_pro.language.make_tile(
    tile_type,
    addr=0x0000,
)
```

- addr is the byte offset within the target Buffer and must be determined at compile time.
- The number of bytes in the address range is automatically derived from the shape and dtype of TileType, and no size parameter is accepted; valid_shape does not change the allocated range.
- The address ranges of multiple Tiles must not overlap unintentionally. When the same address is intentionally reused, the developer must ensure correct access timing.

The following example only creates three UB Tiles and does not include data movement or computation:

```python
import pypto_pro.language as pl

tt = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
)

# A single Tile occupies 64 * 128 * 2 = 16384 bytes, that is, 0x4000 bytes.
tile_a = pl.make_tile(tt, addr=0x0000)
tile_b = pl.make_tile(tt, addr=0x4000)
tile_out = pl.make_tile(tt, addr=0x8000)
```

When make_tile is used, the Tile does not carry the TileGroup rotation and mutex metadata used by auto_mutex. Cross-Pipe dependencies require the developer to explicitly synchronize based on the actual data flow. For API details, see [pypto_pro.language.make_tile](../../../../api/pro_api/SIMD-API/resource_management/make_tile.md).

### Address Alignment

Different buffers have different basic alignment requirements for the Tile start address:

| Buffer | Address Alignment Requirement |
|:---|:---|
| UB, L1 Buffer | 32 bytes |
| L0A Buffer, L0B Buffer | 512 bytes |
| L0C Buffer | 64 bytes |
| L0A_MX Buffer, L0B_MX Buffer | 32 bytes |

In addition to the basic address alignment, the actual space occupied by a Tile is also affected by the alignment of dtype, layout, fractal, and shape. For fractal layouts such as NZ and ZN, the address range cannot be estimated based solely on the number of logical elements.

The quantized coefficient Tile addresses in the L0A_MX Buffer and L0B_MX Buffer must also be equal to the Tile addresses of the corresponding L0A Buffer and L0B Buffer shifted right by 4 bits, respectively.

## Using make_tile_group to Create Rotating Tiles

pypto_pro.language.make_tile_group creates a group of Tiles with the same specification at once, and uniformly manages address layout, rotation positions, and mutex metadata:

```python
group = pypto_pro.language.make_tile_group(
    type=tile_type,
    addrs=0x0000,
    mutex_ids=[0, 1],
)
```

TileGroup can be used for single buffering, double buffering, and N-buffering:

| Configuration | Description |
|:---|:---|
| mutex_ids=[0] | Contains one Tile, with no actual rotation effect |
| mutex_ids=[0, 1] | Two Tiles are used alternately, that is, double buffering or ping-pong buffering |
| mutex_ids=[0, 1, 2, ...] | Triple buffering and deeper N-buffering |
| mutex_ids=None, depth=N | Creates N Tiles, but does not provide the mutex metadata required for automatic synchronization |

### Configuring Tile Addresses

addrs supports two forms:

```python
import pypto_pro.language as pl


# Pass a single base address: tiles are laid out consecutively according to the size of a single slot.
continuous = pl.make_tile_group(
    type=tt,
    addrs=0x0000,
    mutex_ids=[0, 1],
)

# Pass an address list: explicitly specify the address of each tile.
separate = pl.make_tile_group(
    type=tt,
    addrs=[0x0000, 0x10000],
    mutex_ids=[2, 3],
)
```

When an address list is passed, the number of addresses must match the TileGroup depth. When a single base address is used, the framework calculates the subsequent addresses based on the slot size occupied by a single tile.

### Configuring mutex and depth

A mutex ID is the number of an on-chip buffer mutual-exclusion synchronization resource, used to coordinate access to Tiles by different Pipes. Configuring mutex_ids for a TileGroup specifies the ID used by each Tile in the group. After enabling @pypto_pro.language.jit(auto_mutex=True), the framework automatically inserts synchronization operations based on these IDs. You can also create a rotating Tile by passing only depth without mutex_ids, but in this case the framework does not allocate mutex IDs. Even if auto_mutex=True is enabled, mutex synchronization is not automatically inserted for these Tiles, and the developer must ensure the access timing themselves.

- Each item in mutex_ids corresponds to one Tile, and the ID value range is [0, 31].
- Each Tile can be bound to one ID or to a non-empty list of IDs. Multiple IDs bound to the same Tile must not be duplicated.
- When mutex_ids is non-empty and depth is not specified, the TileGroup depth is derived from len(mutex_ids).
- When mutex_ids is None or an empty list, depth must be explicitly specified, and the developer must ensure the access timing.

When allocating IDs, prioritize assigning different IDs to each Tile, both within the same TileGroup and across different TileGroups. For example, double buffering uses mutex_ids=[0, 1], and another Tile that also participates in the pipeline uses mutex_ids=[2]. This avoids unrelated Tiles waiting for each other due to shared IDs, which better leverages pipeline parallelism. When IDs are insufficient, prioritize reusing the same ID for Tiles whose usage periods do not overlap and that do not participate in the pipeline simultaneously. Avoid sharing an ID between two Tiles that may be used at the same time.

If you also explicitly call pypto_pro.language.system.mutex_lock/mutex_unlock, keep their IDs separate from the IDs used by automatic synchronization. Only when you confirm that the usage periods of the two are completely non-overlapping can you reuse the same ID, to avoid repeatedly acquiring a mutex resource that has not been released. The TileGroup itself does not actively perform locking or unlocking at runtime. For the complete parameter description, see [pypto_pro.language.make_tile_group](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md).

## Accessing Tiles in a TileGroup

A TileGroup provides rotation-based access and subscript-based access:

| Access Method | Advances Rotation Position | Description |
|:---|:---:|:---|
| group.next() | Yes | Advances the rotation position and returns the next Tile; consecutive calls select Tiles cyclically |
| group.current() | No | Returns the current Tile |
| group.previous() | No | Returns the Tile preceding the current Tile |
| group[i] | No | Returns the Tile with subscript i, where i can be a compile-time integer or a runtime integer expression; a runtime subscript must be within the range [0, depth) |

The initial rotation position of a TileGroup is the last Tile, so the first call to next() returns Tile 0; before next() is called, current() returns the last Tile.

```python
import pypto_pro.language as pl


double_buffer = pl.make_tile_group(
    type=tt,
    addrs=0x0000,
    mutex_ids=[0, 1],
)

current_tile = double_buffer.current()
next_tile = double_buffer.next()
previous_tile = double_buffer.previous()
first_tile = double_buffer[0]
```

When calling next() alternately on the same TileGroup, ensure that the call order matches the pipeline iteration of the Kernel. When only a fixed Tile needs to be accessed, use current() or subscript access to avoid unintentionally changing the rotation position.

## Setting the Valid Shape of a Tile

When the Tensor shape is not divisible by the Tile shape, only some elements of the boundary Tile are valid. When valid_shape is omitted in TileType, the dynamic mode is used by default, which is equivalent to valid_shape=[-1, -1], so there is no need to explicitly specify it for the tail block:

```python
import pypto_pro.language as pl


tail_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
)
```

When some dimensions need to represent a fixed valid size, valid_shape can be explicitly specified, for example, [64, -1]. After obtaining a specific Tile, use pypto_pro.language.set_validshape to set the valid range of this access. Each dimension must be greater than 0 and cannot exceed the corresponding dimension of the Tile:

```python
import pypto_pro.language as pl


tile = tile_group.next()
pl.set_validshape(tile, [valid_rows, valid_cols])
```

You can also call set_validshape directly on a TileGroup to set all Tiles in the group in batch:

```python
import pypto_pro.language as pl


pl.set_validshape(tile_group, [valid_rows, valid_cols])
```

The valid shape usually needs to be set before data is moved in, to ensure that subsequent movements use the correct boundary. Whether compact and pad need to be configured depends on the memory space, layout, and specific data path. For the number of tail blocks, valid shape, and multi-core task allocation, see [Tail Block Processing](tiling/multi_core_tiling.md#tail-block-processing).

## Reinterpreting a Tile View

pypto_pro.language.reinterpret can reinterpret an existing Tile or TileGroup with a new dtype, shape, or layout:

```python
import pypto_pro.language as pl


new_tile = pl.reinterpret(
    tile,
    dtype=pl.DT_INT32,
    shape=[64, 64],
    layout=pl.ND,
)
```

The new object shares the same Buffer address and size as the original Tile. This operation does not move data, nor does it perform numeric or layout conversion. The storage footprint of the new shape and dtype must not exceed the address range bound to the original Tile, and the caller must also ensure that the physical data in the buffer conforms to the new layout.

reinterpret requires at least one of dtype, shape, or layout to be specified. When dtype is explicitly specified, shape must also be specified. The new view does not inherit the valid_shape of the original Tile; the runtime valid range must be reset through pypto_pro.language.set_validshape.

After reinterpret is performed on a TileGroup, the new and old TileGroups share the same set of buffer rotation states. Calling next() on either one affects the other. When independent rotation is required, create the TileGroups separately. For detailed constraints, see [pypto_pro.language.reinterpret](../../../../api/pro_api/SIMD-API/resource_management/reinterpret.md).

## Related Content

For the vector computation APIs on Tile, see [Tile Computation](vector_computation/tile_computation.md). For cube data movement and computation, see [Cube Computation](cube_computation.md). For the cross-Pipe manual synchronization APIs, see [Synchronization API](../../../../api/pro_api/SIMD-API/synchronization/index.md).
