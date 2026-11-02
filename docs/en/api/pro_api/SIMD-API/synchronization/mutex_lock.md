# pypto_pro.language.system.mutex_lock

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:29:08.603Z pushedAt=2026-10-28T11:57:29.173Z nl=2 -->

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

Obtains the buffer mutex resource corresponding to mutex_id on the specified pipe. If the resource has not been released, the current pipe waits until it can obtain the resource. This API is used to prevent multiple pipes from accessing the same on-chip buffer simultaneously.

## Prototype

```python
pypto_pro.language.system.mutex_lock(
    *,
    pipe: PipeType,
    mutex_id: Union[int, Scalar],
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| pipe | Input | Enumeration value of pypto_pro.language.PipeType. It must be a specific pipe among MTE1/MTE2/MTE3/V/M/S/FIX. PipeType.ALL is not allowed. |
| mutex_id | Input | Python integer, a constant expression that evaluates to an integer, or an integer-typed runtime Scalar expression. The value range of a static ID is [0, 31], and bool is not accepted. The runtime value of a dynamic ID must be within [0, 31]. |

## Constraints

- mutex_lock must be used in pairs with mutex_unlock on the same pipe and with the same mutex_id, and mutex_lock must be called before mutex_unlock.
- On the same pipe, when a previous mutex_lock has not been released by the corresponding mutex_unlock, the same mutex_id must not be obtained again. Otherwise, the second acquisition keeps waiting and causes a deadlock. Manually and automatically generated mutex operations must not repeatedly obtain the same unreleased ID on the same pipe.
- mutex_lock and mutex_unlock corresponding to the same mutex_id must not be nested, regardless of whether the pipes of each group of operations are the same. When using automatic mutex, nesting with explicit mutex operations on the same ID must also be avoided.
- When multiple groups of mutex_lock and mutex_unlock with the same mutex_id are used consecutively on the same pipe, it is not guaranteed that each group of operations on that pipe completes in order. When the previous operation in the same pipeline must complete before the next operation is executed, use the corresponding single-pipeline barrier API first. If the current pipeline has no corresponding single-pipeline barrier API, use [pypto_pro.language.system.bar_all](bar_all.md).
- mutex_lock and mutex_unlock must be placed on symmetric control flow paths to ensure that every acquired mutex resource is released.
- auto_mutex=True automatically generates mutex operations only for Tiles that carry mutex metadata. Explicitly called mutex_lock is still retained, so automatic and manual synchronization can be used in the same Kernel.
- For regular single-buffer, double-buffer, and N-buffer scenarios, use [make_tile_group](../resource_management/make_tile_group.md) together with auto_mutex=True. Use mutex_lock and mutex_unlock only when you need precise control over the locking pipe and insertion position.

## Return Value

None

## Examples

The following Kernel computes out = x + x when auto_mutex=False. The input UB uses mutex ID 0 to constrain the access order of MTE2 and V, and the output UB uses mutex ID 1 to constrain the access order of V and MTE3. Each mutex_lock is followed by a corresponding mutex_unlock on the same pipe.

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
