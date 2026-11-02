# Performance Tuning

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-27T01:48:01.524Z pushedAt=2026-10-27T08:22:11.117Z nl=2 -->

The performance of PyPTO Pro operators is affected by factors such as multi-core task partitioning, data movement efficiency, on-chip data reuse, and the parallelism between computation and data movement pipelines. Performance tuning should be based on collected data: first analyze the Kernel time consumption, AI Core pipeline, memory access, and task timeline to identify the bottleneck, and then select the corresponding optimization method.

Performance optimization should be based on the premise that functionality and precision are correct. After each adjustment of the Tile Shape, synchronization relationship, or multi-core partitioning, precision regression must be performed first, and then the performance gain is verified.

## Performance Tools and Scenarios

| Tool | Scenarios | Primary Outputs or Analysis Content |
|---|---|---|
| torch_npu.profiler | Controls the collection scope in Python code, suitable for collecting performance data of PyPTO Pro kernels. | Kernel time consumption, AI Core pipeline metrics, and task timeline. |
| [msOpProf](https://gitcode.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md) | On-board performance analysis of a single operator and intra-core pipeline collection. | Operator time consumption, AI Core pipeline, memory, cache, resource conflicts, and intra-core pipeline diagram. |

`msOpProf` is provided by the `msopprof` executable in the CANN package, and its command API is consistent with `msprof op`. The following uses the `msprof op` command for illustration.

## Establishing a Performance Baseline

Comparable performance data must come from identical running conditions. Before collection, fix the input Shape, data type, TilingKey, `block_dim`, Stream, and synchronization position, and complete JIT warm-up. When measuring pure Kernel time consumption, do not include input allocation, random number generation, Host-to-Device copy, and result verification. Use the same collection configuration before and after optimization, and eliminate incidental fluctuations through multiple measurements.

Time consumption measured under different measurement scopes cannot be directly compared. When recording results, note the measurement scope, and before operator delivery, verify the end-to-end performance of the actual call chain.

## Collecting Data with torch_npu.profiler

`torch_npu.profiler` can collect the time consumption and AI Core pipeline metrics of PyPTO Pro kernels in Python code. The following example assumes that the script under test has already defined `kernel`, `block_dim`, and `args`:

```python
import torch
import torch_npu


profiler_output = "./profiling_output"
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)

# Complete JIT warm-up before formal collection.
kernel[None, block_dim](*args)
torch.npu.synchronize()

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False,
    record_shapes=False,
    profile_memory=False,
    experimental_config=experimental_config,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        profiler_output, analyse_flag=True
    ),
):
    kernel[None, block_dim](*args)
    torch.npu.synchronize()
```

| File | Items to Check |
|---|---|
| kernel_details.csv | Locate the target kernel in the Name or Type column, and view Duration(us), Block Dim, and AI Core pipeline metrics. |
| trace_view.json or trace_result.json | View the time relationships among the application layer, CANN layer, and NPU tasks. |

## Collecting Data with msOpProf

Use the `msprof op` command to collect on-device performance data for the operators in the PyPTO Pro test script:

```bash
msprof op python3 test_example.py
```

Before formal collection, complete JIT warm-up in the test script. Use `--aic-metrics` to select the required AI Core metrics; when collecting multiple groups of metrics, keep the input and running conditions unchanged. For the collection commands and parameter configuration of the intra-core pipeline, see [*msOpProf User Guide*](https://gitcode.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md). Common metrics are as follows:

| Metric | Primary Check Content | Locatable Issues |
|---|---|---|
| PipeUtilization | Time consumption or proportion of Cube, Vector, Scalar, and data movement pipelines. | Dominant pipeline, pipeline idle, and insufficient overlap. |
| ArithmeticUtilization | Execution of Cube and Vector compute instructions. | Redundant computation or insufficient compute unit utilization. |
| Memory, MemoryL0, MemoryUB | Read/write bandwidth of storage levels such as GM, L1 Buffer, L0A Buffer, L0B Buffer, L0C Buffer, and UB. | Movement-bound, insufficient effective bandwidth, or insufficient on-chip reuse. |
| L2Cache | L2 Cache access and hit status. | Insufficient Cache reuse caused by unreasonable data tiling or access order. |
| ResourceConflictRatio | UB bank conflict, bank group, and other resource conflicts. | On-chip data layout or concurrent access conflicts. |

The collection results are saved in the `OPPROF_*` directory. The output files depend on the `--aic-metrics` configuration. Common files are as follows:

| File | Check Content |
|---|---|
| OpBasicInfo.csv | Basic information such as operator name, type, Task Duration(us), Block Dim, Device ID, and AI Core frequency. |
| PipeUtilization.csv | Time consumption and proportion of compute and movement pipelines on each Core. |
| ArithmeticUtilization.csv | Time consumption and proportion of Cube and Vector compute instructions. |
| Memory.csv, MemoryL0.csv, MemoryUB.csv | Read/write bandwidth of GM and on-chip storage at each level. |
| L2Cache.csv | L2 Cache hit status. |
| ResourceConflictRatio.csv | UB bank conflict, bank group, and other resource conflicts. |
| trace.json | Instruction-level timeline of Scalar, MTE, Cube, Vector, Fixpipe, and other pipelines within each Core. After opening with Trace Viewer or MindStudio Insight, you can check pipeline bubbles, dependencies, and parallelism. |
| visualize_data.bin | Visualization data file used by MindStudio Insight. |
| dump/ | Intermediate files such as raw data and Kernel binaries generated during collection, which usually do not need to be viewed directly. |

## Performance Data and Pipeline Graph Analysis

After collection is complete, locate the bottleneck in the following order:

1. Identify the target Kernel from `kernel_details.csv` or `OpBasicInfo.csv`, and compare the Kernel time consumption across multiple samples. Performance conclusions should be based on independent, stable multi-round measurement results.
2. Compare the time consumption and proportion of the Cube, Vector, and data movement pipelines to identify the dominant pipeline. The pipeline proportion only reflects time coverage; when evaluating compute efficiency and bandwidth utilization, you must also consider instruction throughput, bandwidth, and blocking metrics.
3. Estimate the theoretical time consumption based on the compute volume, data movement volume, and target hardware specifications, and then compare it with the measured results to determine whether the bottleneck stems from the workload itself or from insufficient hardware utilization.
4. Use `trace.json` to check whether the critical pipelines are continuous, and whether data movement, Vector, and Cube can overlap effectively.
5. Select an optimization direction based on the analysis results, such as multi-core partitioning, Tile Shape, on-chip reuse, or multi-buffering. Adjust only one category of key factors per round, and after completing precision regression, re-measure using the same criteria.

The goal of pipeline optimization is to keep the primary pipeline that determines Kernel performance executing continuously during the steady-state phase, and to hide the overhead of other pipelines within the primary pipeline as much as possible. When the total time is primarily limited by a particular compute or data movement pipeline, the Kernel reaches the bound state of that pipeline. Cube-computation Kernels typically target Cube bound; vector-computation or data-movement-intensive Kernels may reach Vector bound or MTE bound respectively. Determining the bound cannot rely solely on the cumulative proportion; you must also confirm whether the primary pipeline is continuous within the main execution interval, and whether other pipelines sufficiently overlap with it.

In Figure 1, both the Cube and Vector pipelines show obvious gaps, and the compute tasks fail to execute continuously. At this point, the Kernel has not yet reached a stable compute bound. You should focus on checking the Tile granularity, data dependencies, synchronization waits, and the overlap between data movement and computation.

**Figure 1 Gaps in the Cube and Vector pipelines**

![Cube and Vector pipelines both have obvious gaps](../../../figures/pro/pro_pipeline_with_gaps.png "Cube and Vector pipelines both have obvious gaps")

In Figure 2, the Cube pipeline remains continuous in the steady state, with MTE, Fixpipe, and Vector tasks executing in parallel with Cube computation. At this point, the Kernel reaches the Cube bound, that is, the performance upper limit of the Cube pipeline.

**Figure 2 Cube pipeline reaches the bound state**

![Cube pipeline executes continuously and forms a Cube bound](../../../figures/pro/pro_pipeline_cube_bound.png "Cube pipeline executes continuously and forms a Cube bound")

## Multi-Core Partitioning Optimization

PyPTO Pro uses [`pypto_pro.language.get_block_idx()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_idx.md) and [`pypto_pro.language.get_block_num()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_num.md) to partition multi-core tasks. The partitioning scheme must fully cover the computation range, prevent multiple cores from writing to the same output region, and ensure load balancing across cores.

- `block_dim` should not exceed the number of tasks that can be executed in parallel; otherwise, idle work units are produced. For the constraints on the number of logical cores, see [Kernel Function](../development/kernel_function.md#meaning-and-setting-of-blockdim).
- The workload of each core should be as balanced as possible. Avoid concentrating tail blocks or high-overhead branches on a few cores.
- For regular 2D tiles, number them linearly first, and then assign them across cores in a strided manner to reduce tail load imbalance.
- An output region should be written by a single core. When cross-core reduction is required, use an explicit and supported synchronization and reduction scheme.
- Select the number of cores separately for small shapes and large shapes. A single `block_dim` usually cannot cover the optimal configuration for all scenarios.

Using `block_dim=1` during debugging helps verify the logic, but the target number of cores must be restored for performance testing.

## Tile and On-Chip Memory Optimization

Tile Shape simultaneously affects the per-computation workload, on-chip memory usage, loop count, tail block ratio, and data movement efficiency. The following factors must be considered comprehensively when selecting a Tile:

- Capacity limits of various on-chip buffers.
- The actual byte count determined jointly by the data type and Tile Shape.
- Alignment requirements for data movement and computation instructions.
- The tail block ratio and the padding overhead beyond valid data.
- The number of times the same input data is reused on-chip.
- The multiplication of total memory usage after multi-buffering.

A larger Tile reduces the number of loops and instruction dispatches, but occupies more on-chip memory, which may prevent double buffering from being enabled or even reduce parallelism. A smaller Tile has lower resource usage, but increases the loop count, the number of data movements, and the tail block processing overhead. The final configuration should be determined through actual measurement based on the target Shape.

GM access should remain contiguous and aligned, and should be merged into large-granularity data movements whenever possible. Data that needs to be reused should reside on-chip when capacity permits, avoiding repeated loading from GM. Intermediate results that can be directly consumed by subsequent computation should not be written back to GM and then reloaded.

## Pipeline and Multi-Buffering Optimization

[`pypto_pro.language.make_tile_group`](../../../../api/pro_api/SIMD-API/resource_management/make_tile_group.md) allocates multiple rotating tiles for the same logical data: `current()` returns the current tile, and `next()` switches to the next one. After enabling `@pypto_pro.language.jit(auto_mutex=True)`, the compiler inserts intra-core synchronization based on the mutex information of the tiles, thereby building a double-buffering or N-buffering pipeline.

During optimization, focus on the following checks:

- Whether loading the next block of data can overlap with the Vector or Cube computation of the current block.
- Whether writing back the current result can overlap with subsequent computation.
- Whether each rotating tile uses an independent, non-conflicting mutex number.
- Whether the number and positions of `next()` calls are consistent with the production and consumption order of the data.
- Whether manual synchronization is correctly paired, and whether it duplicates the protection of the same dependency as the automatic mutex.
- Whether the pipeline is correctly drained at the end, and whether the last result has been fully written back.

Increasing the number of buffers raises on-chip memory usage and scheduling overhead. Only when double buffering still cannot hide the critical latency should you consider increasing the number of buffer stages, and confirm the benefit through Profiling.

## Cube and Vector Collaborative Optimization

Cube computation is described as Cube tasks through [`pypto_pro.language.section_cube()`](../../../../api/pro_api/SIMD-API/controlflow/section_cube.md), and vector computation is described as Vector tasks through [`pypto_pro.language.section_vector()`](../../../../api/pro_api/SIMD-API/controlflow/section_vector.md). For mixed Kernels, you should overlap Cube computation, Vector pre-/post-processing, and DMA data movement as much as possible, and reduce unnecessary data format conversion and cross-storage-layer data movement.

- For Cube computation, check the Tile Shape in the M, N, and K directions, the left/right cube layout, the transpose mode, and the L0A Buffer/L0B Buffer loading format.
- When the reduction length is large, perform block-wise accumulation in the L0C Buffer, and then convert and write back as needed.
- Vector pre-/post-processing should overlap with the Cube pipeline as much as possible to avoid forming a global serial stage.
- When multiple fine-grained Vector operations are executed consecutively and the additional overhead is significant, you can use Vector Function to express register-level computation after confirming the bottleneck, thereby reducing intermediate Tile reads and writes.
- After adjusting the parallel relationship between Cube and Vector, you must re-check the mutex and actual data dependencies. Synchronization must not be eliminated at the cost of correctness.

## TilingKey and Compile-Time Specialization

TilingKey is used to distinguish a limited number of compile-time configurations with clearly different execution paths. For example, dedicated kernels can be generated for aligned and unaligned paths, different algorithm modes, or a small number of fixed layouts, eliminating ineffective branches in hot loops.

Do not expand runtime shapes with a large value range into TilingKeys one by one. An excessive number of keys increases binary size, compilation time, cache usage, and testing cost. A new TilingKey should be added only when the specialization benefit is clear and the candidate set is controllable.

For common shapes, a regular, branch-free main path can be designed, with tail blocks and low-frequency scenarios placed in separate branches. Branches that depend on runtime data should be reduced in hot loops, while necessary boundary checks are retained.

## Result Verification and Delivery Check

After tuning is complete, perform the following checks:

- All target shapes, data types, layouts, TilingKeys, and `block_dim` pass the accuracy test.
- Boundary scenarios such as the minimum shape, non-aligned shape, tail block, idle Core, and maximum resource usage are covered.
- The baseline and the optimized version are compared under the same measurement conditions, and multiple rounds of results are saved.
- Profiling data can explain the performance changes, and accidental fluctuations are not used as the basis for optimization conclusions.
- Debug APIs and extra synchronization used only for locating issues have been removed.
- Performance verification is completed for both the JIT direct invocation path and the actual aclnn invocation path.
- The binary size, the number of TilingKeys, and the first compilation time remain within an acceptable range.
