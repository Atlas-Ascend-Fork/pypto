# pypto_pro.language.system.mutex_unlock

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:29:57.491Z pushedAt=2026-10-28T11:57:29.176Z nl=2 -->

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

Releases the buffer mutex resource corresponding to mutex_id on the specified pipe, allowing other pipes waiting for this resource to continue execution. This API is used together with mutex_lock.

## Prototype

```python
pypto_pro.language.system.mutex_unlock(
    *,
    pipe: PipeType,
    mutex_id: Union[int, Scalar],
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| pipe | Input | A pypto_pro.language.PipeType enumeration value. It must be one specific pipe among MTE1/MTE2/MTE3/V/M/S/FIX; PipeType.ALL is not allowed. |
| mutex_id | Input | A Python integer, a constant expression that evaluates to an integer, or a runtime Scalar expression of integer type. The value range of a static ID is [0, 31], and bool is not accepted; the runtime value of a dynamic ID must be within [0, 31]. |

## Constraints

- mutex_unlock must be used in pairs with a preceding mutex_lock on the same pipe and with the same mutex_id.
- Do not omit mutex_unlock, and do not call mutex_unlock without having acquired the corresponding mutex resource.
- For common constraints such as nesting, reuse, control flow, and automatic mutex, see [pypto_pro.language.system.mutex_lock](mutex_lock.md#constraints).

## Return Value

None

## Examples

The following Kernel computes out = x + x when auto_mutex=False. Each mutex_unlock releases the mutex resource previously acquired on the same pipe with the same mutex ID.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=False)
def mutex_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x4000)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(tile_x, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=0)
        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=1)
        pl.add(tile_out, tile_x, tile_x)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=1)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=1)
        pl.store(out, tile_out, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=1)
```
