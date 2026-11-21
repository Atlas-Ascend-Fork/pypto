# pypto_pro.language.system.dcci

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

Performs data cache clean and invalidate (DCCI) on the data cache corresponding to the specified address. A typical use case is to clear stale cache copies that the current execution unit may hold when sharing data across cores or pipelines, so that the published data can be observed in subsequent access.

DCCI only processes the cache status and is not equivalent to pipeline synchronization, cross-core events, or memory barriers. The caller must still ensure that the producer's writes are complete and uses a synchronization API that matches the communication protocol to establish the sequence.

## Prototype

```python
pypto_pro.language.system.dcci(
    target: Union[Tensor, Tile],
    offset: Optional[Union[Sequence[int], int]] = None,
    *,
    cache_line: CacheLine = pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE,
    dst: DcciDst = pypto_pro.language.DcciDst.AUTO,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| target | Input| GM tensor variable, or a tile allocated in the UB.|
| offset | Input| (Optional) Element offset, in the unit of the target.dtype element. The GM tensor supports the offset list/tuple of each dimension, or the linear offset represented by an integer constant or a runtime integer scalar expression. The length of the list/tuple must be the same as the number of tensor dimensions. The framework converts the linear offset based on the tensor stride. The UB tile supports only the linear offset represented by an integer constant or a runtime integer scalar expression. The default value is 0, which means that the target start address is used. The offset must not cause the valid address to exceed the allocated range of the target.|
| cache_line | Input| (Optional) Compile-time enumerated values of [pypto_pro.language.CacheLine](../basic_data_structures/CacheLine.md). The default value is pypto_pro.language.CacheLine.ENTIRE_DATA_CACHE. The cache line is 64 bytes. The address of SINGLE_CACHE_LINE does not need to be aligned downwards by the user. The hardware operation includes the cache line that contains the address. If the data spans multiple cache lines, make calls by line or use ENTIRE_DATA_CACHE.|
| dst | Input| (Optional) Compile-time numerated values of [pypto_pro.language.DcciDst](../basic_data_structures/DcciDst.md). The default value is pypto_pro.language.DcciDst.AUTO. For details about the meaning and applicable targets of each enumerated value, see DcciDst.|

## Constraints

- cache_line and dst must be determined at the compile time, and cannot be dynamically selected by the scalar or tensor at the runtime.
- SINGLE_CACHE_LINE covers only one 64-byte cache line. When the address range [addr, addr + bytes) is processed, the number of calls must be at least the number of cache lines covered by the range. You cannot call the API only once for the start address.
- ENTIRE_DATA_CACHE is used for the entire data cache, and the offset does not reduce its effective range. This mode has higher overhead than the single cache line operation.
- DCCI is not a synchronization primitive. After a producer writes data through a pipeline such as MTE3, the data must be synchronized to the S pipeline before DCCI is executed or the release flag is issued. The consumer must also complete the corresponding cross-core waiting before performing cache processing and data reads. The specific event ID and synchronization mode are determined by the upper-layer communication protocol.
- Frequently executing DCCI on the entire cache will cause significant performance loss. If the shared data range is known, it is recommended that the data be processed in 64-byte cache lines.

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
    # Clean and invalidate the 64-byte cache line where the start element of inp is located.
    pl.system.dcci(inp, [0, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    inp = torch.ones([16, 16], device=device, dtype=torch.float32)
    out = torch.ones([16, 16], device=device, dtype=torch.float32)

    dcci_gm_kernel(inp, out)
    torch.npu.synchronize()

    # DCCI only processes the cache status and does not modify data.
    assert torch.allclose(out.cpu(), inp.cpu())
    print("dcci done")
```

### Full Cache Invalidation

```python
# When DCCI is performed on the entire data cache, the overhead is greater than that of a single cache line operation. The offset does not reduce the effective range.
pl.system.dcci(inp, cache_line=pl.CacheLine.ENTIRE_DATA_CACHE)
```

### Using DCCI with Synchronization

When performing DCCI on the UB tile, you must use sync_src/sync_dst to establish the pipeline sequence (DCCI is not a synchronization primitive).

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
        # DCCI (S) can be executed only after the load (MTE2) is complete.
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.dcci(t, cache_line=pl.CacheLine.SINGLE_CACHE_LINE)
        # The data is stored (MTE3) after DCCI (S) is completed. Otherwise, the store may read the old data before the data becomes invalid.
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t, [0, 0])
```
