# pypto_pro.language.system.set_cross_core

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:31:38.672Z pushedAt=2026-10-28T11:57:29.180Z nl=2 -->

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

Inter-core synchronization control API, used together with [pypto_pro.language.system.wait_cross_core](wait_cross_core.md). event_id specifies the inter-core synchronization event. Each event ID corresponds to a counter with an initial value of 0. After the preceding instructions in the pipeline specified by pipe complete, executing pypto_pro.language.system.set_cross_core increments the corresponding counter by 1. When the paired pypto_pro.language.system.wait_cross_core is executed, if the counter is 0, the subsequent instructions in the pipeline specified by pipe are blocked; if the counter is greater than 0, the counter is decremented by 1 and the subsequent instructions continue to execute.

The API supports full-core synchronization of the same type of AIC or AIV, synchronization between AIVs within the same AI Core, and synchronization between AIC and AIV within the same AI Core. The cores participating in synchronization and the signal pairing method are determined by sync_mode.

## Prototype

```python
pypto_pro.language.system.set_cross_core(
    *,
    pipe: PipeType,
    event_id: Union[int, Scalar],
    sync_mode: CrossCoreSyncMode = pypto_pro.language.CrossCoreSyncMode.INTRA_BLOCK,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| pipe | Input | [pypto_pro.language.PipeType](../basic_data_structures/PipeType.md) enumeration value, indicating the hardware pipeline where the signal is sent. SET takes effect only after the preceding instructions in this pipeline complete. When sync_mode is INTER_BLOCK, INTER_SUBBLOCK, or INTRA_BLOCK, the value can be M, V, MTE1, MTE2, MTE3, or FIX; S and ALL are not supported. When sync_mode is UNICAST_BLOCK, S is also allowed, but ALL is still not supported. This value can differ from the pipe of the paired pypto_pro.language.system.wait_cross_core. |
| event_id | Input | Inter-core synchronization event ID. Supports a Python integer constant or a runtime integer Scalar expression. A Python integer constant can currently only be 0 to 15. For a dynamic expression, the caller must ensure that the runtime value is valid: 0 to 15 for INTER_BLOCK, INTER_SUBBLOCK, and INTRA_BLOCK; 0 to 15 on the AIV side and 0 to 31 on the AIC side for UNICAST_BLOCK. In UNICAST_BLOCK, 0 to 15 sent by AIV0 pairs with 0 to 15 waited by AIC, and 0 to 15 sent by AIV1 pairs with 16 to 31 waited by AIC; 0 to 15 sent by AIC pairs with 0 to 15 waited by AIV0, and 16 to 31 sent by AIC pairs with 0 to 15 waited by AIV1. The counter corresponding to each event ID ranges from 0 to 15. When the signal of the same event is not consumed by WAIT, sending SET more than 15 times consecutively triggers an exception and interrupts execution. For INTER_BLOCK, INTER_SUBBLOCK, and INTRA_BLOCK, before the same core reuses an event ID or uses the same event ID for a different synchronization mode, all SET and WAIT operations of that event ID in the previous synchronization process must be completed. When used together with [pypto_pro.language.system.sync_all](sync_all.md#constraints), avoid the event IDs occupied by HARD mode; when automatic pipeline scheduling is used, also avoid conflicts with the event IDs it allocates. When the same core sends multiple SETs consecutively, the order in which different event IDs take effect is not guaranteed; when there is a dependency, complete the previous group of SET/WAIT first. |
| sync_mode | Input | Inter-core synchronization mode, used to specify the cores participating in synchronization and the pairing method of SET/WAIT signals. Must use the same mode as the paired pypto_pro.language.system.wait_cross_core. For values, see [pypto_pro.language.CrossCoreSyncMode](../basic_data_structures/CrossCoreSyncMode.md). |

## Constraints

- A pypto_pro.language.system.wait_cross_core matching the current call must exist, and all cores participating in synchronization must be able to reach the synchronization point; otherwise, a deadlock may occur.
- When INTER_BLOCK is used and multiple streams or multiple operators execute concurrently, if the total number of cores requested by the concurrent operators exceeds the number of physical cores, and at least two concurrent operators use inter-core synchronization, some cores may fail to reach the synchronization point because they are not scheduled, causing a deadlock. Ensure that the cores required by each synchronizing operator can execute simultaneously.

## Return Value

None

## Examples

### INTER_BLOCK

```python
with pl.section_vector():
    # Pre-operation on this AIV.
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=0,
        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
    )
```

### INTER_SUBBLOCK

```python
with pl.section_vector():
    # AIV0 and AIV1 each perform their pre-operations.
    pl.system.set_cross_core(
        pipe=pl.PipeType.V,
        event_id=1,
        sync_mode=pl.CrossCoreSyncMode.INTER_SUBBLOCK,
    )
```

### INTRA_BLOCK

```python
with pl.section_vector():
    # AIV0 and AIV1 send signals after completing their pre-operations.
    pl.system.set_cross_core(
        pipe=pl.PipeType.MTE3,
        event_id=2,
        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
    )
```

### UNICAST_BLOCK

```python
with pl.section_vector():
    if pl.get_subblock_idx() == 0:
        # Only AIV0 sends the signal.
        pl.system.set_cross_core(
            pipe=pl.PipeType.S,
            event_id=15,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
        )
```

### Complete Kernel Examples

```python
import pypto_pro.language as pl


@pl.jit()
def cross_core_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat = pl.make_tile(
        pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                    layout=pl.NZ),
        addr=0x10000)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x0000)
        tile_y = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x2000)
        tile_sum = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x4000)
        tile_nz = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                        layout=pl.NZ),
            addr=0x6000)

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(v1_mat, tile_nz, [off, 0])
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=2,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )

    with pl.section_cube():
        rhs_mat = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                        layout=pl.NZ),
            addr=0x0000)
        v1_left = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                        layout=pl.NZ),
            addr=0x0000)
        rhs_right = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                        layout=pl.ZN),
            addr=0x0000)
        c_l0c = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                        layout=pl.NZ, fractal=1024),
            addr=0x0000)

        pl.load(rhs_mat, rhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(rhs_right, rhs_mat)
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE1,
            event_id=2,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.move(v1_left, v1_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, v1_left, rhs_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])
```
