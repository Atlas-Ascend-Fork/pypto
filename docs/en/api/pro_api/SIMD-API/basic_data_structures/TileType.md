# pypto_pro.language.TileType

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T12:51:32.358Z pushedAt=2026-10-28T11:57:28.656Z nl=2 -->

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

Describes the specification of a tile, including its shape, data type, target memory space, and layout. It is used together with [pypto_pro.language.make_tile](../resource_management/make_tile.md) or [pypto_pro.language.make_tile_group](../resource_management/make_tile_group.md) to allocate the actual buffer.

TileType itself does not allocate memory; it is only a specification descriptor. The actual buffer is created through make_tile (single tile) or make_tile_group (multi-tile rotation).

## Prototype

```python
pypto_pro.language.TileType(
    *,
    shape: Sequence[int],
    dtype: DataType,
    target_memory: MemorySpace = pypto_pro.language.MemorySpace.Vec,
    valid_shape: Optional[Sequence[int]] = None,
    layout: Optional[TensorLayout] = None,
    fractal: Optional[int] = None,
    pad: Optional[Union[TilePad, int]] = None,
    compact: Optional[int] = None,
) -> TileType
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| shape | Input | Size of each dimension of the tile, for example, [64, 128]. A compile-time constant integer list of length 2, where each dimension must be a positive integer. Only 2D tiles are supported. Alignment and fractal layout constraints are checked by the specific API that uses this TileType. |
| dtype | Input | Data type of the elements, of the [pypto_pro.language.DataType](DataType.md) type. |
| target_memory | Input | Target memory space, an enumeration value of [pypto_pro.language.MemorySpace](MemorySpace.md), defaulting to UB. |
| valid_shape | Input | Optional. Valid shape (for handling tail blocks or non-full blocks). A compile-time constant integer list or None (default).<br>- Specific integers (for example, [32, 64]): The valid shape is determined at compile time.<br>- None (default): Equivalent to [-1, -1] (dynamic mode).<br>- [-1, -1]: The valid shape is set dynamically at runtime, used together with pypto_pro.language.set_validshape. |
| layout | Input | Optional. Layout, an enumeration value of [pypto_pro.language.TensorLayout](TensorLayout.md) or None (default). If not specified, the default value is determined by the memory space (see [Constraints](#constraints)). For L1 Buffer, NZ is used when layout is not specified, and ZN can also be explicitly specified. When the data type is DT_UINT64 or DT_INT64, ND can also be explicitly specified. BiasTable Buffer defaults to and supports only ND. L0A_MX Buffer supports only ZZ, and L0B_MX Buffer supports only NN. The available layouts for UB are constrained by the specific Tile API. |
| fractal | Input | Optional. Fractal size, an integer or None (default). For FP32/INT32 tiles in L0C Buffer, it is automatically set to 1024 when not specified. For DT_FP8E8M0 tiles with ZZ or NN layout in L1 Buffer and tiles in L0A_MX Buffer/L0B_MX Buffer, fractal=32 is the default and only supported value. Other value requirements are determined by the specific Tile API. |
| pad | Input | Optional. Padding mode, supporting the [pypto_pro.language.TilePad](TilePad.md) type or the int type. When the int type is used, the value range is [0, 3]:<br>- 0: No padding.<br>- 1: Pad with zeros.<br>- 2: Pad with the maximum value.<br>- 3: Pad with the minimum value.<br>An invalid value raises ValueError, and an invalid type raises TypeError. |
| compact | Input | Optional. Compact layout mode of the Tile buffer, describing how the layout is interpreted in the move, rearrangement, and matrix computation paths of the Tile. It does not change the data type, nor does it replace valid_shape in describing the actual valid region. The values are as follows:<br>- **0 or None**: Compact mode is not enabled. In this case, [move](../memory_data_movement/move.md) from L1 Buffer to L0A Buffer/L0B Buffer moves data entirely according to the physical shape, and set_validshape does not take effect on this move path.<br>- **1**: Uses the normal compact mode. Data is laid out contiguously in the valid space where valid_shape is aligned upward to the fractal granularity, and the remaining space declared by the Tile is idle at the tail. For details, see [Data Layout in Normal Compact Mode](#data-layout-in-normal-compact-mode). It is typically used in tail block scenarios, together with set_validshape. Whether it is configured on L1 Buffer has no effect on the result. When used together with phase, if a tail block exists, L0C Buffer must be configured with compact=1; otherwise, the system may hang. compact does not fill the invalid region; when filling is required, use the pad parameter or fillpad. compact does not change the allocated size of the buffer but only changes the data layout within the buffer. Therefore, when concatenating multiple tiles into a wider operand, pay special attention to the address offset. For details, see [Tail Block Processing for Multi-Tile Concatenation](#tail-block-processing-for-multi-tile-concatenation).<br>- **2**: Uses the RowPlusOne compact mode. It is configured only in UB Tile and is used to avoid bank conflicts when moving data with this Tile as the source. For details, see [Data Layout in RowPlusOne Compact Mode](#data-layout-in-rowplusone-compact-mode). In NZ format, each fractal column reserves one extra physical row, which serves only as a placeholder and does not participate in computation. Therefore, when applying for the physical shape of the Tile, this extra row must be included, and when using the data, configure the actual valid number of rows through set_validshape. The same applies to ZN format, where one extra column is reserved. |

## Constraints

### layout Default Value Description

If layout is not specified, the default value is automatically selected based on the memory space:

| Memory Space | Default layout |
|---|---|
| UB | No constraint |
| L1 Buffer | NZ |
| L0A Buffer | NZ |
| L0B Buffer | ZN |
| L0C Buffer | NZ |
| BiasTable Buffer | ND |
| Fixpipe Buffer | ND |
| L0A_MX Buffer | ZZ |
| L0B_MX Buffer | NN |


### Data Layout in Normal Compact Mode

```text
  Example: In the L0A Buffer, the physical shape of a Tile is [64, 64], the data type is FP16, valid_shape is [8, 24], and the format is NZ. Therefore:
        1. The fractal granularity D is 16 rows x 16 columns, totaling 512 bytes (0x200).
        2. The valid space for actually moving data is [ceil16(8), ceil16(24)] = [16, 32], totaling 2 fractals.

  Without compact configured: data is laid out according to shape=[64,64]

     0   16   32   48   64  <- column
   0 ┌────┬────┬────┬────┐
     │ D1 │ D2 │    │    │   D1, D2 = two valid data fractals
  16 ├────┼────┼────┼────┤
     │    │    │    │    │   Stride between two adjacent columns = 64 rows = 4 fractals = 0x800
  32 ├────┼────┼────┼────┤   D1 start address 0x000, D2 start address 0x800
     │    │    │    │    │
  48 ├────┼────┼────┼────┤
     │    │    │    │    │
  64 └────┴────┴────┴────┘

  With compact=1 configured: data is laid out contiguously within the valid space [16,32]

     0   16   32
   0 ┌────┬────┐          Stride between two adjacent columns = 16 rows = 1 fractal = 0x200
     │ D1 │ D2 │          D1 start address 0x000, D2 start address 0x200
  16 └────┴────┘
     ┌─────────────────────┐
     │ Remaining declared space, idle │   ← Buffer is still allocated by shape
     └─────────────────────┘
```

### Data Layout in RowPlusOne Compact Mode

```text
  Example: The physical shape of a Tile in the UB is [64, 64], in NZ format.

  With compact=1: fractal columns are placed end to end.

     ┌────┬────┬────┐
     │ D1 │ D2 │ D3 │   64 rows per column
     └────┴────┴────┘
       ↑    ↑    ↑
     First-address stride = 64 rows; each column starts on the same bank.


  With compact=2: one extra row is reserved per column.

     ┌─────┬─────┬─────┐
     │ D1  │ D2  │ D3  │   64 rows of data per column
     │ ─── │ ─── │ ─── │ ← One extra row at the end of each column (placeholder)
     └─────┴─────┴─────┘
       ↑     ↑     ↑
     First-address stride = 65 rows; each column starts one row later than the previous one, landing on a different bank.
```

## Return Value

Returns a Tile type descriptor object. This object only stores the Tile's attributes such as shape, dtype, memory space, and layout, and does not allocate on-chip memory. Pass it to make_tile or make_tile_group to create an actual Tile.

## Examples

TileType only describes the tile specification. The actual buffer is created by make_tile or make_tile_group.

### Default UB Tile

```python
import pypto_pro.language as pl

# target_memory defaults to MemorySpace.Vec.
vec_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16)
```

### Tiles in Each Cube Memory Space

```python
import pypto_pro.language as pl

# Tile in the L0A buffer: The default layout of Left is NZ, which can also be specified explicitly.
left_type = pl.TileType(
    shape=[64, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Left, layout=pl.NZ,
)

# Transposed fractal tile in the L1 buffer: Mat supports explicitly specifying ZN.
mat_zn_type = pl.TileType(
    shape=[128, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
)

# Tile in the L0C buffer: When fractal is not specified for FP32 Acc, it automatically defaults to 1024.
acc_type = pl.TileType(
    shape=[64, 128], dtype=pl.DT_FP32,
    target_memory=pl.MemorySpace.Acc,
)
```

### Dynamic Tail Block Tile

```python
import pypto_pro.language as pl

# valid_shape=[-1, -1]: Set the actual valid shape at runtime through set_validshape.
# compact=1: Interpret the dynamic tail block compactly by the valid window in the load, move, and matmul paths.
tail_type = pl.TileType(
    shape=[64, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1], compact=1,
)
```

### compact = 1 Row-Direction Tail Block

```python
import os
import pypto_pro.language as pl
import torch

DEVICE = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
TILE = 128
M_TAIL = 72

@pl.jit(auto_mutex=True)
def matmul_m_tail(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[M_TAIL, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ, valid_shape=[-1, -1],
        ),
        addrs=0x0,
        mutex_ids=[0],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    # L0A Buffer: matmul always reads data in compact layout, and compact=1 makes move write with the same stride.
    a_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
            layout=pl.NZ, valid_shape=[-1, -1], compact=1,
        ),
        addrs=0x0,
        mutex_ids=[2],
    )
    b_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN
        ),
        addrs=0x0,
        mutex_ids=[3],
    )
    # L0C Buffer: matmul always writes data in compact layout, and compact=1 makes store read with the same stride.
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ, fractal=1024, valid_shape=[-1, -1], compact=1,
        ),
        addrs=0x0,
        mutex_ids=[4],
    )
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(al, [M_TAIL, TILE])
        pl.set_validshape(ac, [M_TAIL, TILE])
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])

if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    torch.manual_seed(42)
    a = torch.randn([TILE, TILE], device=DEVICE, dtype=torch.float16)
    b = torch.randn([TILE, TILE], device=DEVICE, dtype=torch.float16)
    out = torch.zeros([M_TAIL, TILE], device=DEVICE, dtype=torch.float32)

    matmul_m_tail(a, b, out)
    torch.npu.synchronize()

    ref = torch.matmul(a[:M_TAIL, :].float(), b.float())
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### Tail Block Processing for Multi-Tile Concatenation

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit(auto_mutex=True)
def matmul_wide_alias(
    x: pl.Tensor[[64, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    out: pl.Tensor[[40, 128], pl.DT_FP32],
):
    x_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ
        ),
        addrs=0x00000,
        mutex_ids=[0],
    )
    y_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ
        ),
        addrs=0x10000,
        mutex_ids=[1],
    )
    x_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=0x0000,
        mutex_ids=[2, 3],
    )
    # Alias: same base address, used as the left operand of matmul.
    x_wide = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ
        ),
        addrs=0x0000,
        mutex_ids=[[2, 3]],
    )
    y_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, layout=pl.ZN
        ),
        addrs=0x0000,
        mutex_ids=[4],
    )
    c_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ, fractal=1024, valid_shape=[-1, -1],
        ),
        addrs=0x0000,
        mutex_ids=[5],
    )

    with pl.section_cube():
        cur_x = x_l1.current()
        cur_y = y_l1.current()
        br = y_l0b.current()
        ac = c_l0c.current()

        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.move(br, cur_y)

        # The Tiles in the two L0A Buffers do not set compact=1 or set_validshape; data is fully transferred to ensure the data layout in the concatenated L0A Buffer is contiguous.
        al_1 = x_l0a.next()
        pl.move(al_1, cur_x, [0, 0])
        al_2 = x_l0a.next()
        pl.move(al_2, cur_x, [0, 64])

        al = x_wide.next()
        pl.matmul(ac, al, br)

        # The L0C Buffer does not set compact=1; store transfers data in the matmul write format; set_validshape controls the amount of data transferred.
        pl.set_validshape(ac, [40, 128])
        pl.store(out, ac, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.randn([64, 128], device=device, dtype=torch.float16)
    y = torch.eye(128, device=device, dtype=torch.float16)
    out = torch.zeros([40, 128], device=device, dtype=torch.float32)

    matmul_wide_alias(x, y, out)
    torch.npu.synchronize()

    ref = x[:40, :].float()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    print(f"wide alias matmul passed, max diff = {(out - ref).abs().max().item()}")
```
