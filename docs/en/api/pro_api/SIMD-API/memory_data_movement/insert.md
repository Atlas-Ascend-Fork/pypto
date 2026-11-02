# pypto_pro.language.insert

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:00:07.542Z pushedAt=2026-10-28T11:57:28.703Z nl=2 -->

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

Embeds a smaller source Tile into a larger destination Tile at the row and column position specified by offset=[row, col]. It is recommended to use [pypto_pro.language.move](move.md) instead of the insert API.

## Prototype

```python
pypto_pro.language.insert(
    dst_tile: Tile,
    src_tile: Tile,
    /,
    offset: List[int],
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dst_tile | Output | Destination operand, of the Tile type. For supported data types and fractal combinations, see [Constraints](#constraints). |
| src_tile | Input | Source operand, of the Tile type. For supported data types and fractal combinations, see [Constraints](#constraints). |
| offset | Input | Position offset, of the List[int] type, with a length of 2 and the format [row, col]. The top-left corner of the source Tile is aligned to the offset position of the destination Tile. offset[0] is the row offset of the destination Tile, and offset[1] is the column offset. row and col must be non-negative integers or runtime integer expressions, and the valid region of the source Tile must fall completely within the destination Tile. |

## Constraints

- Data type and fractal constraints:

  | Source → Destination | Fractal Requirements | Data Type Requirements |
  |---|---|---|
  | UB → UB | ND → ND, NZ → NZ. | The source and destination must be identical. DT_INT8, DT_INT32, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported. |
  | UB → L1 Buffer | The source supports ND and NZ. The fractal of the destination is not checked. | The source and destination must be identical. DT_INT8, DT_INT32, DT_FP16, DT_BF16, DT_FP32, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2 are supported. |
  | L0C Buffer → UB | NZ → ND, NZ → DN, NZ → NZ. | DT_FP32 → DT_FP32/DT_FP16/DT_BF16 and DT_INT32 → DT_INT32 are supported. |
  | L0C Buffer → L1 Buffer | NZ → NZ. | DT_FP32 → DT_FP32/DT_FP16/DT_BF16 and DT_INT32 → DT_INT32 are supported. |

  When the source operand is in UB in the ND format and the destination operand is in L1 Buffer in the NZ or ZN format, avoid direct data transfer. First transfer the source operand to another UB in the NZ format using pypto_pro.language.move, and then transfer it to the destination operand. For the correct usage, see [Concatenating the Computation Results in UB to the L1 Buffer](#concatenating-the-computation-results-in-ub-to-the-l1-buffer).

## Return Value

None

## Examples

### Concatenating the Computation Results in UB to the L1 Buffer

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def insert_matmul_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=0x10000, mutex_ids=[0])

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0000, mutex_ids=[1])
        tile_y_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x2000, mutex_ids=[2])
        tile_sum_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x4000, mutex_ids=[3])
        tile_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                             layout=pl.NZ),
            addrs=0x6000, mutex_ids=[4])
        v1_mat = v1_mat_group.current()
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        tile_sum = tile_sum_group.current()
        tile_nz = tile_nz_group.current()

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])

        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)   # ND -> NZ

        pl.insert(v1_mat, tile_nz, [off, 0])   # UB -> L1 NZ2NZ
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[5])
        v1_left_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[6])
        rhs_right_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN),
            addrs=0x0000, mutex_ids=[7])
        c_l0c_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024),
            addrs=0x0000, mutex_ids=[8])
        v1_mat = v1_mat_group.current()
        rhs_mat = rhs_mat_group.current()
        v1_left = v1_left_group.current()
        rhs_right = rhs_right_group.current()
        c_l0c = c_l0c_group.current()

        pl.load(rhs_mat, rhs, [0, 0])
        pl.move(rhs_right, rhs_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)

        pl.matmul(c_l0c, v1_left, rhs_right)

        pl.store(out, c_l0c, [0, 0])
```

### Other Two-Dimensional Offset Scenarios

```python
# Offsets exist in both dimensions.
pl.insert(p_mat_slot, p_f16_back_slot, [TKV // 2, TS_HALF * sub_id])

# Offset only along dimension 0.
pl.insert(v1_mat, tile_nz, [off, 0])

```
