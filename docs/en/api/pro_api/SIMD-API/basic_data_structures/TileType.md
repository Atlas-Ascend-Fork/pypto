# pypto_pro.language.TileType

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

Describes the specifications of a tile, such as the shape, data type, memory space, and layout. It is used together with [pypto_pro.language.make_tile](../resource_management/make_tile.md) or [pypto_pro.language.make_tile_group](../resource_management/make_tile_group.md) to allocate the actual buffer.

TileType itself does not allocate memory. It is only a specification descriptor. The actual buffer is created through make_tile (single tile) or make_tile_group (multiple tiles in rotation mode).

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
    compact: Optional[int] = 1,
) -> TileType
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| shape | Input| Size of each dimension of a tile, for example, [64, 128]. Compile-time constant integer sequence (list or tuple) with a length of 2. The size of each dimension must be a positive integer. Only two-dimensional tiles are supported. The alignment and fractal layout constraints are checked by the specific API that uses TileType.|
| dtype | Input| Data type of the elements. It is an enumerated value of [pypto_pro.language.DataType](DataType.md).|
| target_memory | Input| Target memory space, which is an enumerated value of the [pypto_pro.language.MemorySpace](MemorySpace.md) type. The default value is UB.|
| valid_shape | Input| (Optional) Valid shape (for processing tail blocks or non-full blocks). Compile-time constant integer sequence (list or tuple) or None (default).<br>- Specific integers (for example, [32, 64]): The valid shape is determined at compile time.<br>- None (default): It is equivalent to [-1, -1] (dynamic mode).<br>- [-1, -1]: The valid shape is dynamically set at runtime and used together with pypto_pro.language.set_validshape.|
| layout | Input| (Optional) Layout, which is an enumerated value of [pypto_pro.language.TensorLayout](TensorLayout.md) or None (default). If this parameter is not specified, the default value is used based on the memory space. For details, see [Constraints](#constraints). If the layout of the L1 Buffer is not specified, NZ is used by default. You can also explicitly specify the ZN layout. When the data type is DT_UINT64 or DT_INT64, you can also explicitly specify the ND layout. The BiasTable Buffer supports only the ND layout by default. The L0A_MX Buffer supports only the ZZ layout, and the L0B_MX Buffer supports only the NN layout. The available layouts of the UB are restricted by the specific tile API.|
| fractal | Input| (Optional) Fractal size, which is an integer or None (default). If not specified, it is automatically set to 1024 for FP32/INT32 tiles in the L0C Buffer. For DT_FP8E8M0 tiles with the ZZ or NN layout in the L1 Buffer and the tiles in the L0A_MX Buffer/L0B_MX Buffer, fractal=32 is the default and only supported value. Other values are determined by the specific tile API.|
| pad | Input| (Optional) Padding mode, which is of the [pypto_pro.language.TilePad](TilePad.md) or int type. When the value is of the int type, the value range is [0, 3].<br>- 0: No padding.<br>- 1: Padding with zeros.<br>- 2: Padding with the maximum value.<br>- 3: Padding with the minimum value.<br>If the value is invalid, ValueError is thrown. If the type is invalid, TypeError is thrown.|
| compact | Input| (Optional) Compact layout mode of the tile buffer. It describes the layout interpretation mode of tiles in the data movement, rearrangement, and matrix computation paths. It does not change the data type or replace valid_shape in the description of the valid region. The options are as follows:<br>- **0**: The compact mode is disabled. In this case, the [move](../memory_data_movement/move.md) from the L1 Buffer to the L0A Buffer/L0B Buffer is performed based on the physical shape. The set_validshape parameter is invalid in this movement path.<br>- **1 (default)**: The normal compact mode is used. The data is continuously arranged in the valid space aligned to the fractal granularity of valid_shape. The remaining space declared by the tile is idle at the end. For details, see [Data Layout in Normal Compact Mode](#data-layout-in-normal-compact-mode). This mode is usually used in the tail tile scenario and is used together with set_validshape. Whether it is configured on the L1 Buffer has no impact on the result. If compact is used together with phase and there is a tail tile, compact must be set to 1 for the L0C Buffer. Otherwise, the system may be suspended. compact does not pad invalid regions. If padding is required, use the pad parameter or fillpad. compact does not change the allocated size of the buffer, but only changes the data layout within the buffer. Therefore, when combining multiple tiles into a wider operand, pay special attention to the address offset. For details, see [Tail Block Processing for Multi-Tile Concatenation](#tail-block-processing-for-multi-tile-concatenation).<br>- **2**: The RowPlusOne compact mode is used. It is configured only in the UB tile to avoid bank conflicts when the tile is used as the source for data movement. For details, see [Data Layout in RowPlusOne Compact Mode](#data-layout-in-rowplusone-compact-mode). In the NZ layout, an extra row of physical space is reserved for each fractal column, which is only used as a placeholder and does not participate in computation. Therefore, the physical shape of the tile to be applied for must contain this extra row, and the actual number of valid rows is configured using set_validshape. The same applies to the ZN layout, where an extra column is reserved.|

## Constraints

### Default layout Values

If layout is not specified, the default value is automatically selected based on the memory space.

| Memory Space| Default Layout|
|---|---|
| UB | No constraints|
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
  Assume that the physical shape of the tile in the L0A Buffer is [64, 64], the data type is FP16, valid_shape is [8, 24], and the layout is NZ.
        1. The fractal granularity D is 16 rows × 16 columns, and the total size is 512 bytes (0x200).
        2. The valid space for data movement is [ceil16(8), ceil16(24)] = [16, 32], and there are two fractals in total.

  Explicitly setting compact to 0: Data is arranged based on shape=[64,64].

     0   16   32   48   64  ← Column
   0 ┌────┬────┬────┬────┐
     │ D1 │ D2 │    │    │   D1 and D2 are two valid fractals.
  16 ├────┼────┼────┼────┤
     │    │    │    │    │   Stride between two adjacent columns = 64 rows = 4 fractals = 0x800
  32 ├────┼────┼────┼────┤   The start address of D1 is 0x000, and the start address of D2 is 0x800.
     │    │    │    │    │
  48 ├────┼────┼────┼────┤
     │    │    │    │    │
  64 └────┴────┴────┴────┘

  Setting compact to 1: Data is continuously arranged in the valid space [16, 32].

     0   16   32
   0 ┌────┬────┐          Stride between two adjacent columns = 16 rows = 1 fractal = 0x200
     │ D1 │ D2 │          The start address of D1 is 0x000, and the start address of D2 is 0x200.
  16 └────┴────┘
     ┌─────────────────────┐
     │ The remaining space is idle. │   ← The buffer is still allocated based on the shape.
     └─────────────────────┘
```

### Data Layout in RowPlusOne Compact Mode

```text
  Assume that the physical shape of the UB tile is [64, 64] and the NZ layout is used.

  Setting compact to 1: Fractal columns are placed end to end.

     ┌────┬────┬────┐
     │ D1 │ D2 │ D3 │   64 rows per column
     └────┴────┴────┘
       ↑    ↑    ↑
     The first-address stride is 64 rows, and each column starts on the same bank.


  Setting compact to 2: An extra row is reserved for each column.

     ┌─────┬─────┬─────┐
     │ D1  │ D2  │ D3  │   64 rows of data per column
     │ ─── │ ─── │ ─── │ ← An extra row at the end of each column (placeholder)
     └─────┴─────┴─────┘
       ↑     ↑     ↑
     The first-address stride is 65 rows, and each column starts one row later than the previous one, landing on a different bank.
```

## Return Value

Returns a description object of the tile type. This object only saves the attributes of the tile, such as shape, dtype, memory space, and layout, and does not allocate on-chip memory. It needs to be passed to make_tile or make_tile_group to create an actual tile.

## Examples

TileType only describes the tile specifications. The actual buffer is created through make_tile or make_tile_group.

### Default UB Tile

```python
import pypto_pro.language as pl

# The default value of target_memory is MemorySpace.Vec.
vec_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16)
```

### Tiles in Each Cube Memory Space

```python
import pypto_pro.language as pl

# Tiles in the L0A Buffer: The default layout of Left is NZ, which can also be explicitly specified.
left_type = pl.TileType(
    shape=[64, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Left, layout=pl.NZ,
)

# Transposed fractal tiles in the L1 Buffer: ZN can be explicitly specified for Mat.
mat_zn_type = pl.TileType(
    shape=[128, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
)

# Tiles in the L0C Buffer: If fractal is not specified for FP32 Acc, 1024 is automatically used.
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
    # L0A Buffer: matmul always reads data in the compact layout. When compact is set to 1, data is written in the same stride in the move operation.
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
    # L0C Buffer: matmul always writes data in the compact layout. When compact is set to 1, data is written in the same stride in the store operation
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
            shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, layout=pl.NZ, compact=0
        ),
        addrs=0x0000,
        mutex_ids=[2, 3],
    )
    # Alias: The same base address is used as the left operand of matmul.
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
            layout=pl.NZ, fractal=1024, valid_shape=[-1, -1], compact=0,
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

        # compact is set to 0 for the tiles in the two L0A Buffers, set_validshape is not called, and data is fully moved to ensure that the data in the combined L0A Buffer is continuously arranged.
        al_1 = x_l0a.next()
        pl.move(al_1, cur_x, [0, 0])
        al_2 = x_l0a.next()
        pl.move(al_2, cur_x, [0, 64])

        al = x_wide.next()
        pl.matmul(ac, al, br)

        # compact is explicitly set to 0 for the L0C Buffer, the store operation moves data in the write format of matmul, and set_validshape controls the amount of data to be moved.
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
