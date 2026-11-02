# pypto_pro.language.matmul

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T12:57:59.050Z pushedAt=2026-10-28T11:57:28.692Z nl=2 -->

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

Performs matrix multiplication on lhs_tile and rhs_tile, and writes the result to dst_tile:

```text
dst_tile = lhs_tile × rhs_tile
```

## Prototype

```python
pypto_pro.language.matmul(
    dst_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    bias_tile: Optional[Tile] = None,
    /,
    *,
    phase: Optional[AccPhase] = None,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dst_tile | Output | Destination operand, of the Tile type, stored in the L0C Buffer, with a shape of [M, N] and the NZ layout. When the data type is DT_FP32 or DT_INT32, fractal is automatically set to 1024 if not specified. The valid shape of the tail block can be set through valid_shape or pypto_pro.language.set_validshape, and the valid M and N must be consistent with the actual matrix multiplication result range. For supported data type combinations, see [Constraints](#constraints). |
| lhs_tile | Input | Source operand (left matrix), of the Tile type, stored in the L0A Buffer, with a shape of [M, K] and the NZ layout. K must be consistent with the K dimension of rhs_tile. For supported data type combinations, see [Constraints](#constraints). |
| rhs_tile | Input | Source operand (right matrix), of the Tile type, stored in the L0B Buffer, with a shape of [K, N] and the ZN layout. K must be consistent with the K dimension of lhs_tile. For supported data type combinations, see [Constraints](#constraints). |
| bias_tile | Input | Source operand (optional bias), of the Tile type, stored in the BiasTable Buffer, with a shape of [1, N] and the ND layout (the default and only supported layout). The bias is broadcast along the M dimension, and its data type must be consistent with dst_tile. When this parameter is passed, the bias addition is fused in the Fixpipe stage, eliminating the need to call pypto_pro.language.add separately. It can only be passed as the fourth positional parameter. |
| phase | Input | Optional. Enables the hardware unitFlag mechanism, of the [pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md) type. For how it works with [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md), see [Advanced Cube Computation](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md). |

## Constraints

- Data type constraints:

  | lhs_tile (L0A Buffer) | rhs_tile (L0B Buffer) | dst_tile (L0C Buffer) |
  |---|---|---|
  | DT_INT8 | DT_INT8 | DT_INT32 |
  | DT_FP16 | DT_FP16 | DT_FP32 |
  | DT_BF16 | DT_BF16 | DT_FP32 |
  | DT_FP32 | DT_FP32 | DT_FP32 |
  | DT_HF8 | DT_HF8 | DT_FP32 |
  | DT_FP8E4M3FN or DT_FP8E5M2 | DT_FP8E4M3FN or DT_FP8E5M2 | DT_FP32 |

- matmul with bias_tile initializes the L0C Buffer and can only be used for the first block of K-dimensional blocking; subsequent blocks must use matmul_acc for accumulation.
- For K-dimensional blocking, use matmul for the first block to initialize the L0C Buffer, and use matmul_acc for subsequent blocks to accumulate. When phase is enabled, use pypto_pro.language.AccPhase.Partial for non-final blocks and pypto_pro.language.AccPhase.Final for the final block, in conjunction with the [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md) of pypto_pro.language.store, pypto_pro.language.store_tile, and pypto_pro.language.move.

## Return Value

None

## Examples

### Single matmul (without bias and K-dimensional blocking)

```python
import pypto_pro.language as pl

TILE = 128
M_SIZE = 256
K_SIZE_MM = 128      # K is exactly one Tile, so no block accumulation is required.
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
):
    # L1 double buffering (rotated by next())
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A Buffer, L0B Buffer, and L0C Buffer all use a single-tile group (current())
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):          # M-dimensional blocking
            for j in pl.range(0, N_SIZE, TILE):      # N-dimensional blocking
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                al = a_left.current()
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
```

### matmul with bias (without K-dimensional blocking)

```python
import pypto_pro.language as pl

TILE = 128
M_SIZE = 256
K_SIZE_MM = 128
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_bias_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    bias: pl.Tensor[[1, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP16],
):
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[6])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[7])
    bias_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
        addrs=0x0000, mutex_ids=[8])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[9])

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):
            for j in pl.range(0, N_SIZE, TILE):
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                cur_bias_l1 = bias_l1.next()
                al = a_left.current()
                br = b_right.current()
                bl = bias_l0b.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.load(cur_bias_l1, bias, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.move(bl, cur_bias_l1)             # Move the FP16 data from the L1 buffer into the BiasTable buffer and convert it to FP32.
                pl.matmul(ac, al, br, bl)             # out = A @ B + bias
                pl.store(c, ac, [i, j])
```

### K-Dimensional Blocking Accumulation with Bias

```python
import pypto_pro.language as pl

TILE = 128
K_SPLIT = 384     # Split into 3 TILE blocks.


@pl.jit(auto_mutex=True)
def matmul_k_split_bias_kernel(
    a: pl.Tensor[[TILE, K_SPLIT], pl.DT_FP16],
    b: pl.Tensor[[K_SPLIT, TILE], pl.DT_FP16],
    bias: pl.Tensor[[1, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP16],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[6, 7])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[8, 9])
    bias_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
        addrs=0x0000, mutex_ids=[10, 11])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[12])

    with pl.section_cube():
        ac = acc.current()
        for k in pl.range(0, K_SPLIT, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                # First block: load bias, and matmul overwrites acc = A0@B0 + bias.
                bias_l1_tile = bias_l1.next()
                pl.load(bias_l1_tile, bias, [0, 0])
                bl = bias_l0b.next()
                pl.move(bl, bias_l1_tile)
                pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
            elif k < K_SPLIT - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
```
