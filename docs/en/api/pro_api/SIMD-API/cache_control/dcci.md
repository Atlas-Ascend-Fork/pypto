# pypto_pro.language.system.dcci

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:51:52.927Z pushedAt=2026-10-28T11:57:28.667Z nl=2 -->

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

Performs data cache clean and invalidate (DCCI) on the data cache corresponding to the specified address. A typical use case is to clear stale cache copies that the current execution unit may hold when sharing data across cores or pipelines, so that subsequent accesses can observe the published data.

dcci only handles cache state and is not equivalent to pipeline synchronization, cross-core events, or memory barriers. The caller must still ensure that the producer's writes are complete and use a synchronization API that matches the communication protocol to establish the ordering relationship.

## Prototype

```python
pypto_pro.language.system.dcci(
    target: Union[Tensor, Tile],
    offset: Optional[Union[int, Sequence[int], Expr]] = None,
    *,
    cache_line: CacheLine = pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE,
    dst: DcciDst = pypto_pro.language.DcciDst.AUTO,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| target | Input | GM Tensor variable, or a Tile allocated in UB. |
| offset | Input | Optional, element offset, in units of target.dtype elements. For a GM Tensor, supports a per-dimension offset list/tuple, as well as a linear offset expressed as an integer constant or a runtime integer scalar expression; the length of the list/tuple must match the number of Tensor dimensions, and the framework converts it to a linear offset based on the Tensor stride. For a UB Tile, only a linear offset expressed as an integer constant or a runtime integer scalar expression is supported. Defaults to 0, which means the target start address is used; the offset must not cause the effective address to exceed the allocated range of the target. |
| cache_line | Input | Optional, a compile-time [pypto_pro.language.CacheLine](../basic_data_structures/CacheLine.md) enumeration value, defaulting to pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE. The cache line size is 64 bytes; for SINGLE_CACHE_LINE, the address does not need to be aligned downward by the user, and the hardware operation covers the cache line containing that address. If the data spans multiple cache lines, call it line by line or use ENTIRE_DATA_CACHE. |
| dst | Input | Optional, a compile-time [pypto_pro.language.DcciDst](../basic_data_structures/DcciDst.md) enumeration value, defaulting to pypto_pro.language.DcciDst.AUTO. For the meaning and applicable targets of each enumeration value, see DcciDst. |

## Constraints

- cache_line and dst must be determined at compile time and cannot be dynamically selected by a runtime Scalar or Tensor.
- SINGLE_CACHE_LINE covers only one 64-byte cache line. When processing the address range [addr, addr + bytes), the number of calls must be at least the number of cache lines covered by the range; it cannot be called only once for the start address.
- ENTIRE_DATA_CACHE acts on the entire data cache, and offset does not narrow its scope; this mode has higher overhead than a single-cache-line operation.
- DCCI is not a synchronization primitive. After a producer writes data through a pipeline such as MTE3, it must first synchronize the data to the S pipeline before executing DCCI or publishing a flag. Likewise, a consumer must first complete the corresponding cross-core wait before performing cache processing and data reads. The specific event number and synchronization mode are determined by the upper-layer communication protocol.
- Frequently executing DCCI on the entire cache causes significant performance loss. When the shared data range is known, process it by 64-byte cache lines first.

## Return Value

None

## Examples

### Single Cache Line Invalidation (GM Tensor)

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit()
def dcci_gm_kernel(
    inp: pl.Tensor[[16, 16], pl.DT_FP32],
    out: pl.Tensor[[16, 16], pl.DT_FP32],
):
    # Clean and invalidate the 64-byte cache line at the address of the first element of inp.
    pl.system.dcci(inp, [0, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    inp = torch.ones([16, 16], device=device, dtype=torch.float32)
    out = torch.ones([16, 16], device=device, dtype=torch.float32)

    dcci_gm_kernel(inp, out)
    torch.npu.synchronize()

    # DCCI only processes the cache state and does not modify data.
    assert torch.allclose(out.cpu(), inp.cpu())
    print("dcci done")
```

### Full Data Cache Invalidation

```python
# Perform DCCI on the entire data cache. This incurs higher overhead than a single cache line operation, and offset does not narrow its scope.
pl.system.dcci(inp, cache_line=pl.CacheLine.ENTIRE_DATA_CACHE)
```

### Using DCCI with Synchronization

When performing DCCI on a UB Tile, you must first use sync_src/sync_dst to establish the pipeline ordering (DCCI is not a synchronization primitive):

```python
@pl.jit(auto_mutex=True)
def dcci_ub_kernel(
    inp: pl.Tensor[[16, 16], pl.DT_FP32],
    out: pl.Tensor[[16, 16], pl.DT_FP32],
):
    tt = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    t = pl.make_tile(tt, addr=0x0000)
    with pl.section_vector():
        pl.load(t, inp, [0, 0])
        # dcci (S) can be executed only after load (MTE2) completes.
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.dcci(t, cache_line=pl.CacheLine.SINGLE_CACHE_LINE)
        # store (MTE3) can be executed only after dcci (S) completes; otherwise, store may read stale data from before the invalidation.
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t, [0, 0])
```
