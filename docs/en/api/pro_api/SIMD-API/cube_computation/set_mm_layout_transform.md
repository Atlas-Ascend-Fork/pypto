# pypto_pro.language.system.set_mm_layout_transform

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:58:48.157Z pushedAt=2026-10-28T11:57:28.699Z nl=2 -->

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

Switches the Fixpipe result readout direction in matmul computation. When enabled, Fixpipe reads data from the L0C Buffer along the N direction.

## Prototype

```python
pypto_pro.language.system.set_mm_layout_transform(*, enabled: bool) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| enabled | Input | Fixpipe result readout direction switch, of bool type, which must be determined at compile time. True indicates reading along the N direction, and False indicates reading along the M direction. Use it only when matmul computation and Fixpipe need to access the same L0C Buffer in parallel. |

## Constraints

- Use it only when matmul computation and Fixpipe access the same L0C Buffer in parallel. Set enabled to False after the result is moved out. No call is required when the two run serially or use different L0C Buffers.

## Return Value

None

## Examples

### Switching the Fixpipe Result Readout Direction During K-Dimension Block Accumulation

```python
import pypto_pro.language as pl

TILE_ACC = 128
K_SIZE_ACC = 256


@pl.jit(auto_mutex=True)
def mm_layout_kernel(
    a: pl.Tensor[[TILE_ACC, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE_ACC], pl.DT_FP16],
    c: pl.Tensor[[TILE_ACC, TILE_ACC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        # The current pipeline scheduling requires Fixpipe to read results along the N direction.
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE_ACC):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)
```
