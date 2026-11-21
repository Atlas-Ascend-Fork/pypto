# pypto_pro.language.system.dsb

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

Data synchronization barrier. Sorts the memory access operations that have been issued by the current core and are within the `scope`. The access operations after the barrier start only after all the access operations before the barrier are complete.

It only sorts the memory access operations of the current core, does not synchronize with other cores, and does not wait for the uncompleted transfers on asynchronous pipes such as MTE. The posted store on an asynchronous pipe needs to be pulled into the current pipe by event synchronization (sync_src/sync_dst or the set_flag/wait_flag pair built in pld.notify/pld.barrier), and then sorted by the dsb. Typical scenario: After MTE3 writes data to GM and before writing inter-core signals or calling sync_all, insert `pl.system.dsb(scope=pl.MemDsb.DDR)` to ensure that the write operation reaches the memory system before the signal.

## Prototype

```python
pypto_pro.language.system.dsb(
    *,
    scope: MemDsb = pypto_pro.language.MemDsb.ALL,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| scope | Input| Optional. [pypto_pro.language.MemDsb](../basic_data_structures/MemDsb.md) enumerated value, which specifies the memory access range for barrier sorting. The default value is pypto_pro.language.MemDsb.ALL (sorting all memory accesses). Use DDR when sorting only GM accesses is required to reduce overhead.|

## Constraints

- dsb sorts only the memory accesses of the current core and does not provide inter-core synchronization. For inter-core synchronization, use sync_all or set_cross_core/wait_cross_core. dsb is responsible for the sequence of current core memory accesses on both sides of the barrier.
- Posted stores on asynchronous pipes (MTE*) are not directly waited for by dsb. If "MTE3 writes to GM become visible before subsequent signals" is required, perform MTE3 event synchronization (for example, sync_src/sync_dst or the built-in pair of pld.barrier(flush_store=True)), and then call dsb.
- dsb is issued on the S pipe. Unlike the bar_* series, dsb is not a pipe barrier and does not wait for a pipe queue to be emptied.

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit
def dsb_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x0200)
    pl.load(tile_a, a, [0, 0])
    pl.add(tile_out, tile_a, tile_a)
    pl.store(out, tile_out, [0, 0])

    # After the MTE3 store completes, sort GM accesses (current core) so that subsequent operations (such as cross-core signal writes) can observe the preceding writes.
    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=0)
    pl.system.dsb(scope=pl.MemDsb.DDR)
```

In this example, after the store operation, an MTE3→S event pair is used to drain the MTE3 queue. Then, dsb(DDR) ensures that subsequent memory accesses (signal writes not shown in the example; see the notify implementation in pypto_distribute.h) of the current core are placed after the GM write.
