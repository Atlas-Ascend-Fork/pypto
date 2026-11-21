# pypto_pro.language.system.set_next_task_start

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

SuperKernel is an operator binary fusion technology. This API is used for synchronization between tasks within the SK subkernel function and is used to enable the behavior of "parallel execution with subsequent subkernels" in the Early-Start capability. Instructions following the call site can execute in parallel with **subsequent** subkernel functions. This API is the dual API of [pypto_pro.language.system.wait_pre_task_end](wait_pre_task_end.md).

This API takes effect only when the kernel function is fused into an SK subkernel. The capability bit 2 (pypto_pro.language.SkCap.CAP_2_EARLY_START_SET) must be enabled in the sk_cap parameter of [JIT compilation](../../../../guide/programming_guide/pro/development/compilation_and_execution/JIT_compilation.md), and the early_start option must be enabled on the SK framework or graph compilation side (for example, `super_kernel_optimize_options={"early_start": 1}` of the TorchAir graph backend). If the kernel is unfused or early_start is disabled, the early start configuration is 0, and this API performs no operation.

## Prototype

```python
pypto_pro.language.system.set_next_task_start(
    *,
    force: bool = True,
) -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                                                                                                                                                                                                                                                                    |
| ------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| force | Input     | Optional, compile-time bool constant. The default value is True. If True, `AscendC::SetNextTaskStart<PIPE_MTE3, PIPE_FIX, true>()` is generated. If False, `AscendC::SetNextTaskStart()` is generated. For Ascend 950PR&950DT products, the default pipes for AIV/AIC are PIPE_MTE3/PIPE_FIX, where the forced form writes them out explicitly. In the SK subkernel context, the default (FORCE=false) form is optimized out during compilation. Therefore, the forced form is used by default.|

## Constraints

- This API can be called only in SK subkernel functions. Calling it in common (unfused) Kernels has no effect.
- This API must be used together with the sk_cap parameter of pypto_pro.language.jit: Set pypto_pro.language.SkCap.CAP_2_EARLY_START_SET (to disable DCCI simultaneously, you can write pypto_pro.language.SkCap.CAP_2_EARLY_START_SET | pypto_pro.language.SkCap.CAP_4_DISABLE_DCCI_ALL). If sk_cap contains CAP_1/CAP_2, the framework automatically adds `-D__ASCENDC_SUPERKERNEL_EARLY_START_V3`.
- This API is used together with [pypto_pro.language.system.wait_pre_task_end](wait_pre_task_end.md) in the paired subkernel. They are used on both sides of the same inter-subkernel boundary. The current subkernel calls this API (set), and the adjacent subsequent subkernel calls wait_pre_task_end (wait). During SK graph compilation, synchronization tasks are generated based on the pairing of adjacent subkernels. If no pairing is performed, the synchronization task degrades to empty synchronization, and no overlapping effect is generated.
- It should be placed after the last data transfer of the operator so that the subsequent instructions can be executed in parallel with the subsequent subkernels.
- It must be called only once per core in each TilingKey branch to avoid being called under different conditions within runtime if/for statements. All potentially executed TilingKey branches must invoke this API; otherwise, a mismatch in synchronization instruction counts may cause deadlocks.
- Avoid conflicts with the inter-core synchronization event IDs of [pypto_pro.language.system.sync_all](sync_all.md) and [pypto_pro.language.system.set_cross_core](set_cross_core.md). Early start synchronization internally occupies inter-core synchronization event IDs.
- The value of force must be a compile-time bool constant. If a string, integer, or runtime scalar is passed, the frontend will reject it.

## Return Value

None

## Examples

In this example, pypto_pro.language.system.set_next_task_start is called after the last data transfer of the SIMD (Tile) kernel so that the instructions following the call site can execute in parallel with the subsequent subkernels.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, sk_cap=pl.SkCap.CAP_2_EARLY_START_SET)
def early_start_set_kernel(
    x: pl.Tensor[[1024, 64], pl.DT_FP32],
    out: pl.Tensor[[1024, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x0100)

    with pl.section_vector():
        for row in pl.range(
            pl.get_block_idx(),
            x.shape[0],
            pl.get_block_num() * pl.get_subblock_num(),
        ):
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])

        # It is called after the last data transfer is complete. Subsequent instructions can execute in parallel with the subsequent subkernels.
        pl.system.set_next_task_start(force=True)
```
