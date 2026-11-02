# pypto_pro.language.system.bar_all

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:22:09.559Z pushedAt=2026-10-28T11:57:29.151Z nl=2 -->

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

Waits for the completion of operations previously issued in the V, M, MTE1, MTE2, MTE3, and FIX pipelines within the current AI Core.

## Prototype

```python
pypto_pro.language.system.bar_all() -> None
```

## Parameters

None

## Constraints

- Can be called in the Cube section or the Vector section.
- Synchronizes only all pipelines within the current AI Core, and is not a global barrier across multiple AI Cores. For multi-core synchronization, use [pypto_pro.language.system.sync_all](sync_all.md).
- This API waits for the completion of operations previously issued in all pipelines within the current AI Core, which may affect performance. When only one pipeline needs to be waited on, use the corresponding single-pipeline barrier API.

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit()
def bar_all_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    out: pl.Tensor[[128, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x2000)
    with pl.section_vector():
        for i in pl.range(0, 128, 64):
            pl.system.bar_all()
            pl.load(tile_x, x, [i, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.add(tile_out, tile_x, tile_x)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, tile_out, [i, 0])
```
