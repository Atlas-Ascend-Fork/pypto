# pypto_pro.language.system.sync_all

<!-- md-trans-meta sourceCommit=a93460828ab8a16873c7742fd1be1143030eb11d translatedAt=2026-10-26T14:30:53.492Z pushedAt=2026-10-28T11:57:29.183Z nl=2 -->

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

Establishes an inter-core barrier among multiple AIV cores, multiple AIC cores, or between AIV and AIC cores. Cores participating in the synchronization wait after reaching sync_all until all participating cores in the current round have arrived, and then continue execution.

Uses FFTS hardware synchronization.

## Prototype

```python
pypto_pro.language.system.sync_all(
    *,
    core_type: SyncCoreType = pypto_pro.language.SyncCoreType.MIX,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| core_type | Input | Optional. An enumeration value of [pypto_pro.language.SyncCoreType](../basic_data_structures/SyncCoreType.md) that specifies the core type participating in the barrier. The default value is pypto_pro.language.SyncCoreType.MIX. This parameter does not specify the number of participating cores. |

## Constraints

- All cores participating in the synchronization must execute sync_all the same number of times in the same order. If the loop counts or branch conditions differ, causing some cores to execute sync_all fewer or more times, a deadlock may occur. In MIX mode, the calls on the AIC side and the AIV side must correspond one-to-one.
- Pure Vector Kernels use AIV_ONLY, and pure Cube Kernels use AIC_ONLY. MIX mode requires both the Cube-side AIC and the Vector-side AIV to execute the corresponding sync_all. Calling it on only one side prevents the other side from reaching the barrier, causing the Kernel to time out.
- When multiple streams or multiple operators execute concurrently and the total number of cores requested by the concurrent operators exceeds the number of physical cores, if at least two concurrent operators use sync_all, some cores may fail to reach the barrier because they are not scheduled, resulting in a deadlock. Ensure that the cores required by each synchronizing operator can execute simultaneously.
- sync_all establishes a barrier among the participating cores. When cross-core GM data reads and writes are required before and after the barrier, the corresponding data visibility requirements must also be met.
- When used together with set_cross_core/wait_cross_core, the MIX barriers on both sides must be on the same side of the SET/WAIT pair. It is prohibited for the Cube side to execute sync_all before SET while the Vector side executes WAIT before sync_all; otherwise, a circular wait is formed.
- sync_all occupies inter-core synchronization event IDs: AIV_ONLY occupies 14 on the AIV side, and AIC_ONLY occupies 11 on the AIC side. MIX occupies 11 to 13 on the AIC side and 12 to 13 on the AIV side. In the MIX 1:2 scenario, the AIC also occupies 28 and 29. When used together with set_cross_core/wait_cross_core, these event IDs must not be used for manual inter-core synchronization that has not yet completed.

## Return Value

None

## Examples

The following is a pure Vector Kernel example showing how the AIV_ONLY barrier is placed. Each AIV processes non-overlapping rows by core number, with TileGroup and auto_mutex handling intra-core pipe dependencies. sync_all is placed outside the loop so that all participating AIVs cross the barrier only after this stage ends.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sync_all_kernel(
    x: pl.Tensor[[2048, 64], pl.DT_FP32],
    out: pl.Tensor[[2048, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    input_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    output_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])

    with pl.section_vector():
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            tile_x = input_tiles.next()
            tile_out = output_tiles.next()
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])

        pl.system.sync_all(
            core_type=pl.SyncCoreType.AIV_ONLY,
        )
```

This example uses the AIV_ONLY barrier.

### Core Types

The three core_type invocation methods are as follows. The following snippets are used for the corresponding types of kernels.

```python
# Pure Vector kernel: executed by all participating AIVs.
with pl.section_vector():
    # ... Vector phase computation
    pl.system.sync_all(
        core_type=pl.SyncCoreType.AIV_ONLY,
    )

# Pure Cube kernel: executed by all participating AICs.
with pl.section_cube():
    # ... Cube phase computation
    pl.system.sync_all(
        core_type=pl.SyncCoreType.AIC_ONLY,
    )

# Kernel with both Cube and Vector: AICs and AIVs must reach the same MIX barrier.
with pl.section_cube():
    # ... Cube phase computation
    pl.system.sync_all(
        core_type=pl.SyncCoreType.MIX,
    )

with pl.section_vector():
    # ... Vector phase computation
    pl.system.sync_all(
        core_type=pl.SyncCoreType.MIX,
    )
```
