# pypto_pro.language.system.sync_dst

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:32:23.350Z pushedAt=2026-10-28T11:57:29.190Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not Supported
<!-- end id3 -->

## Description

Used for synchronization between different pipelines within the same AI Core. When the pipeline specified by wait_pipe reaches this API, if the corresponding flag is 0, the operations after this API in wait_pipe are blocked; if the corresponding flag is 1, the flag is cleared and subsequent operations continue. This API must be used in pairs with [pypto_pro.language.system.sync_src](sync_src.md).

## Prototype

```python
pypto_pro.language.system.sync_dst(
    *,
    set_pipe: PipeType,
    wait_pipe: PipeType,
    event_id: Union[int, Scalar],
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| set_pipe | Input | pypto_pro.language.PipeType enumeration value, indicating the source pipeline, that is, the pipeline that sends the synchronization event. A specific pipeline must be specified, and pypto_pro.language.PipeType.ALL is not supported. For supported pipeline combinations, see the relevant constraints. |
| wait_pipe | Input | pypto_pro.language.PipeType enumeration value, indicating the destination pipeline, that is, the pipeline that waits for the synchronization event. A specific pipeline different from set_pipe must be specified, and pypto_pro.language.PipeType.ALL is not supported. For supported pipeline combinations, see the relevant constraints. |
| event_id | Input | Event ID. Supports Python int, a constant expression that evaluates to an integer, or an integer-type runtime Scalar. bool is not supported. The value range is [0, 7]. |

## Constraints

- Must be used in pairs with pypto_pro.language.system.sync_src. The set_pipe, wait_pipe, and event_id of the two APIs must be exactly the same.
- pypto_pro.language.system.sync_src must be called first to send the event, and then pypto_pro.language.system.sync_dst must be called to wait for the event.
- set_pipe and wait_pipe must form a supported intra-core synchronization path on the current execution side.
- For the same set_pipe and wait_pipe combination, the same event_id can be reused only after the previous pypto_pro.language.system.sync_dst completes waiting. Calling pypto_pro.language.system.sync_src consecutively, omitting either paired API, or having inconsistent control flow paths for the two APIs may cause a data race or deadlock.
- When used together with auto_mutex=True, ensure that explicit synchronization and automatic mutex handle distinct dependencies separately to avoid synchronizing the same dependency repeatedly.

### Supported Pipeline Combinations for Vector Segments

| set_pipe | Supported wait_pipe |
|---|---|
| MTE2 | V, MTE3, S |
| MTE3 | V, MTE2, S |
| V | MTE2, MTE3, S |
| S | V, MTE2, MTE3 |

### Supported Pipeline Combinations for Cube Segments

| set_pipe | Supported wait_pipe |
|---|---|
| MTE1 | MTE2, M, MTE3, FIX |
| MTE2 | MTE1, M, MTE3, S, FIX |
| MTE3 | MTE1, MTE2, S |
| M | MTE1, MTE2, FIX, S |
| S | MTE2, MTE3 |
| FIX | M, MTE2, S, MTE3, MTE1 |

### Typical Synchronization Patterns

| Scenario | set_pipe | wait_pipe | Description |
|---|---|---|---|
| Vector computation after load | MTE2 | V | Data is moved from GM into UB or L1 Buffer before vector computation. |
| Store after vector computation | V | MTE3 | Data is moved out to GM after vector computation completes. |
| Load again after store | MTE3 | MTE2 | New data is moved into the same Buffer after the data is moved out to GM. |

## Return Value

None

## Examples

This example moves two FP32 input tensors from GM into UB. pypto_pro.language.system.sync_dst waits for the MTE2-to-V synchronization event sent by pypto_pro.language.system.sync_src, and performs element-wise addition after receiving the event. It then waits for the V-to-MTE3 synchronization event, and writes the result from UB to GM after receiving the event.

```python
import pypto_pro.language as pl


@pl.jit()
def sync_dst_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tt, addr=0x0000)
    tile_b = pl.make_tile(tt, addr=0x4000)
    tile_out = pl.make_tile(tt, addr=0x8000)
    with pl.section_vector():
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile_out, [0, 0])
```
