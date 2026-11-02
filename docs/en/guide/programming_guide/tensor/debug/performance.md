# Performance Tuning

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T02:21:02.366Z pushedAt=2026-10-28T08:56:02.954Z nl=2 -->

## Overview

During the development of fused operators, operator performance optimization is often extremely challenging. This chapter describes how to use PyPTO Toolkit, a visualization tool, to help developers write high-performance fused operators without needing to understand hardware implementation details in depth.

## Overall Process

After completing precision debugging, developers can use PyPTO Toolkit to view the swimlane diagram of the corresponding operator. The swimlane diagram visually displays the actual scheduling and execution process of the compute graph, clearly presenting the execution order and latency information of tasks. Based on this, developers can obtain the initial performance of the written operator, which can also be referred to as out-of-the-box performance.

By observing the swimlane diagram, developers can identify the performance bottlenecks in the current operator implementation. After identifying the bottlenecks, they can adjust the Tiling configuration and adopt different compute graph compilation strategies to obtain the operator swimlane diagram under the new implementation, make further adjustments, and gradually improve operator performance, thereby achieving Man-In-The-Loop (a human-involved optimization process) tuning.

## Performance Tuning Tools

### Collecting Swimlane Diagram Data

Two parallel collection methods are currently supported: **swimlane diagram collection during graph execution** and **AI CPU/AI Core joint collection**. The two methods can be enabled separately without interfering with each other, or enabled simultaneously. If both are enabled, you can observe AI CPU and AI Core profiling data on the same timeline in the swimlane diagram.

#### Method 1: Swimlane Diagram Collection During Graph Execution (`runtime_debug_mode`)

**Applicable scenarios**: Used for routine performance tuning, focusing on subgraph execution order, task latency distribution, AI Core end-to-end latency, and AI Core utilization.

1. Enable performance data collection by configuring the graph execution debug switch via the `debug_options` parameter of the `@pypto.frontend.jit` decorator:

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

2. Execute the use case:

    ```bash
    python3 examples/02_intermediate/operators/softmax/softmax.py
    ```

3. Collection results are output to the current working directory `output/output_<timestamp>` (for details, see "Collection Result File Description").

#### Method 2: AI CPU/AI Core Joint Collection (`DUMP_DEVICE_PERF`)

**Applicable scenarios**: Used to analyze the collaborative relationship between AI CPU scheduling and AI Core execution, and to locate issues such as slow first-task startup and scheduling waits.

**Restrictions**:

- The tool supports a maximum of 200 rounds of data collection and screen printing. The excess part will be truncated.
- Currently, only data from 20 devTask builds can be collected. When the number of devTask builds (related to the `stitch_function_max_num` configuration, where each stitch corresponds to one devTask build) exceeds 20, the excess perf data will be truncated, and the following warning will appear in the log:

    ```txt
    Dev task num larger than: 20, the excess part will not be recorded
    ```

1. Enable it through an environment variable:

    ```bash
    export DUMP_DEVICE_PERF=true
    ```

2. Execute the use case:

    ```bash
    python3 examples/02_intermediate/operators/softmax/softmax.py
    ```

3. Collection results are output to the current working directory `output/output_<timestamp>` (for details, see "Collection Result File Description").

4. After all data is written to disk, manually run the **analyze** command to view the AI CPU/AI Core data summary table:

    ```bash
    python tools/scripts/machine_perf_trace.py analyze output/output_<timestamp>/machine_trace_perf_data_0.json
    ```

    ![](../../../figures/tensor/dump_device_perf.png "AI CPU/AI Core data summary table")

### Collection Result File Description and Parameter Explanation

#### Output Directory Artifacts

Files related to the swimlane diagram in the `output/output_timestamp` directory:

| File | Purpose |
|---|---|
| `machine_trace_perf_data*.json` | Raw profiling data of the Machine component |
| `tilefwk_L1_prof_data_*.json` | L1-level profiling data of the Machine component |
| `merged_swimlane.json` | IDE visual integrated swimlane diagram |
| `machine_runtime_operator_trace*.json` | AI CPU/AI Core swimlane diagram (joint timeline) |

> `machine_trace_perf_data*.json` and `tilefwk_L1_prof_data_*.json` can be used to determine whether underlying data collection is successful. `merged_swimlane.json` and `machine_runtime_operator_trace*.json` are used for PyPTO Toolkit display.

#### Parameter Explanation

##### CTRL AI CPU

| Phase | Description |
|---|---|
| **DEV_TASK_BUILD** | Time taken for assembling the compute graph compilation result into a device-executable DevTask (stitch phase) |
| **Post-process** | Time taken for the finalization after all DevTasks are built, until the Control AI CPU exits |
| **Total run time** | The complete run time of the Control AI CPU from startup to exit |

##### SCHED AI CPU

| Phase | Description |
|---|---|
| **ALLOC_THREAD_ID** | Time taken for Schedule AI CPU thread allocation |
| **INIT** | Time taken for Schedule AI CPU initialization |
| **CORE_HAND_SHAKE** | Time taken to establish the communication handshake between Schedule AI CPU and AI Core |
| **DEV_TASK_RCV** | Time taken to receive DevTask from Control AI CPU |
| **Post-process** | Time taken for the finalization after all DevTasks are dispatched and executed (including synchronous stop, waiting for AI Core to exit, etc.), until Schedule AI CPU exits |
| **Total run time** | The complete run time of Schedule AI CPU from startup to exit |

##### AICORE

| Phase | Description |
|---|---|
| **INIT** | The initial overhead after AI Core starts, including initialization, handshake with Schedule AI CPU, and waiting to receive the first compute task |
| **End-to-End time** | The time window during which compute tasks are actually executed on all AI Cores, from the earliest start to the latest completion |
| **Post-process** | The trailing overhead after the last compute task completes, until all AI Cores exit |
| **Total run time** | The complete run time of all AI Cores from startup to exit |

### Viewing Swimlane Diagram Data

1. View the AI Core execution end-to-end latency and AI Core utilization via the terminal:

    **Figure 1**  Viewing AI Core Perf information

    ![](../../../figures/tensor/aicore_perf_summary.png "AI Core Perf information")

2. View the swimlane diagram via the PyPTO Toolkit plugin.

    Right-click the corresponding JSON file and choose "Open with PyPTO Toolkit" from the pop-up menu.

    **Figure 2**  Viewing the swimlane diagram

    ![](../../../figures/tensor/view_swimlane_graph.png "View swimlane diagram")

    ![AI CPU/AI Core swimlane diagram](../../../figures/tensor/machine_runtime_operator_trace_0.png)

    The diagram shows the execution order and latency information of tasks, helping developers analyze performance bottlenecks.

### Collecting PMU Data

The PMU (Performance Monitoring Unit) is a critical hardware module in modern processors, specifically designed to monitor and analyze processor performance. The PMU contains multiple programmable counters, each capable of monitoring one or more types of events.

**Step 1: Select Collection Mode**

Select the collection mode via environment variables:

```bash
export PYPTO_PROF_PMU_EVENT_TYPE=<group_id>
```

The value range of `group_id` is `[1, 2, 4, 5, 6, 7, 8]`, and the default value is `2`. The modes supported by PMU are as follows:

![](../../../figures/tensor/PMU_event.png "PMU supported modes")

**Step 2: Collect Data**

After adapting the compilation macros, collect PMU data via the `msprof` command:

```bash
msprof --task-time=l3 [--output=<data storage path>] python xxx.py
```

Where:

- `l3`: Enable the PMU collection switch.
- `--output`: Specifies the output path for PROF artifacts. By default, they are stored in the project root directory.

**Step 3: Parse Data**

After PMU data collection is complete, the data is stored in the `output/PROF*/device_*/data/` directory. Based on the selected `PYPTO_PROF_PMU_EVENT_TYPE`, run the parsing script:

```bash
python tools/profiling/tilefwk_pmu_to_csv.py -p PROF_xxx/device_x/data -pe=$PYPTO_PROF_PMU_EVENT_TYPE --arch [dav_2201, dav_3510]
```

After parsing is complete, the `tilefwk_prof_pmu.csv` file is generated in the project root directory.

### PMU Trace

PMU Trace is used for intra-core pipeline analysis and is an important tool for in-depth operator performance tuning. After this capability is enabled, developers can intuitively observe the timing arrangement and overlap of hardware pipelines (such as MTE2, MTE3, Vector, and Cube) when a kernel is executed on the AI Core. By analyzing the pipeline arrangement, you can identify idle waits between data movement and computation, and determine whether each pipeline is fully utilized, so as to adjust the Tiling configuration or computation orchestration strategy in a targeted manner and improve operator performance.

**Restrictions**:

- Applicable Products:
  <!-- npu="950" id1 -->
  - Ascend 950PR&950DT products: Supported
  <!-- end id1 -->
  <!-- npu="A3" id2 -->
  - Atlas A3 products: Not supported
  <!-- end id2 -->
  <!-- npu="910b" id3 -->
  - Atlas A2 products: Not supported
  <!-- end id3 -->
- Currently only single-operator collection is supported. Full-network scenarios are not supported.
- Currently, data can be collected from a maximum of 6 cores.

#### Enabling PMU Trace

Enable via the `enable_pmu_trace` parameter in `codegen_options`. After it is enabled, codegen inserts `bisheng::cce::mark_stamp` instrumentation points at the beginning and end of the CCE code corresponding to each leaf function, and assigns a PMU ID to each, which is used to locate the corresponding kernel in the intra-core pipeline.

**PMU ID Generation Rules**

PMU IDs are generated by `CodeGenCloudNPU::GenPMUId()` (`codegen_cloudnpu.cpp`) to identify different kernel functions.

Generation method: `PMU ID = [main block marker] + last three digits of funcHash`, and then the result is taken modulo 4096 as the stamp value.

| Component | Rule | Example |
|---|---|---|
| Main Block Marker | When `ctx.isMainBlock == true`, prefix `"1"` is added; otherwise, empty | Main block: `"1"`, non-main block: none |
| Last Three Digits of funcHash | Take the last three characters of `subFunc.GetFunctionHash()` | `"123"` |

**Constraint**: The BiSheng compiler has a stamp value upper limit of 4096. To distinguish the main block from the tail block, only the last three digits of funcHash are currently used. When the number of subgraphs exceeds approximately 1000, PMU IDs across kernels may collide.

Two configuration methods are supported:

**Method 1: Configure in the `@pypto.frontend.jit` decorator**

```python
@pypto.frontend.jit(
    codegen_options={"enable_pmu_trace": True}
)
```

**Method 2: Configure via the `set_codegen_options` API**

```python
pypto.set_codegen_options(enable_pmu_trace=True)
```

#### Collecting PMU Trace Data

After enabling PMU Trace, you also need to enable instruction-level collection via `msprof` at runtime:

```bash
msprof --instr-profiling=on [--output=<data storage path>] python xxx.py
```

Where:

- `--instr-profiling=on`: Enables instruction-level profiling collection.
- `--output`: Manually specifies the output path. Defaults to the current working directory.

#### Collection Result

After collection is complete, the data file is stored in `PROF_*/mindstudio_profiler_output/msprof_*.json`. Developers can load this file in PyPTO Toolkit to perform intra-core pipeline analysis. In the intra-core pipeline view of PyPTO Toolkit, each pair of `mark_stamp` is displayed as a spike marker on the pipeline, and NOP padding creates identifiable intervals on the timeline. By matching the same PMU ID, developers can locate the pipeline segments on the timeline to the corresponding leaf function, thereby analyzing the timing arrangement and utilization of each pipeline (MTE2, MTE3, Vector, Cube, etc.) within that compute phase.

## Out-of-the-Box Performance Tuning

The initial operator performance is most closely related to the loop writing style and the TileShape setting. This chapter describes how to use the relevant APIs to directly achieve good out-of-the-box performance during the initial operator writing process.

### Correctly Choosing the Loop Pattern

Since subgraphs between different root functions cannot be stitched, and subgraph stitching is a key means for PyPTO to optimize performance, the core principle of loop optimization is: **increase the size of root functions and reduce their number**.

#### Using Python for Loops on Static Axes

The pypto.loop method expands the current axis into different root functions through loop unrolling. Therefore, for loops on static axes, use Python for loops instead of PyPTO loop.

   ```python
   # ✅ Recommended: Use Python for for static axes
   for i in range(batch_size):
       result[i] = process(data[i])

   # ❌ Avoid: Use PyPTO loop for static axes
   for i in pypto.loop(batch_size, name="LOOP_1", idx_name="i"):
       result[i] = process(data[i])
   ```

For tuning results, see section 3.3.1 in [GDR Operator Cases](./performance_case_GDR.md).

#### Using PyPTO Loop for Dynamic Axes and Properly Configuring View

When an operator involves dynamic shapes, the dimension value range of dynamic axes is often wide, requiring loop-based processing. In this case, pay attention to the parameter configuration of the view. The selected shape range must not be too small; otherwise, it will limit the configurable range of subsequent TileShape, resulting in too little computation per loop iteration, an increased number of loop iterations, and potentially additional repeated data transfer for some operations. For example, in cube multiplication, an overly small TileShape causes a large number of root functions to transfer the same left or right cube repeatedly. An example of configuring view TileShape to 128 is as follows:

   ```python
   # Recommended: Use loop + unroll for dynamic axes
   bsz, h = x.shape
   b = 128
   b_loop = (bsz + b - 1) // b
   for b_idx in pypto.loop(b_loop, name="LOOP_1", idx_name="b_idx"):
       b_valid = (bsz - b_idx * b).min(b)
       x_view = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
       # Matmul
       pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
       y = pypto.matmul(x_view, W)
   ```

#### Merging Loops Whenever Possible

Check whether the operator code has loop blocks that can be merged. They should be merged to enlarge the root function. For example, the two loops below can be merged, thereby increasing the possibility of merging Operation1 and Operation2 computations to reduce redundant data transfer of y.

```python
bsz = x1.shape[0]
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
       out_1 = Operation1(x1[b_idx, :], y)
for b_idx in pypto.loop(bsz, name="LOOP_2", idx_name="b_idx"):
       out_2 = Operation2(x2[b_idx, :], y)
```

#### Using [loop_unroll](../../../../api/tensor_api/controlflow/pypto-loop_unroll.md) When the Dynamic Axis Has a Wide Range

When an operator uses dynamic shapes with a wide shape range, consider using loop_unroll instead of the loop API. loop_unroll functions similarly to loop but adds the unroll_list parameter to support multiple unrolling modes. For example, when a dynamic axis in an operator needs to generalize and support a shape range of 1 to 64k, specifying a single dynamic axis split size is difficult to meet the requirements. If the split size is too large, small-shape scenarios will introduce many empty computation tasks with an actual computation size of 0, increasing latency. If the split size is too small, large-shape scenarios will have too many loop iterations, affecting overall performance. With loop_unroll, regardless of whether the shape is large or small, the framework selects an appropriate tier or combination based on the unroll_list parameter, avoiding redundant computation and keeping the number of loop iterations under control, thereby achieving better performance.

Note the following points when using it:

- As the number of tiers configured via the unroll_list parameter increases, the number of tasks to be processed during compilation also multiplies, leading to longer compilation time. Therefore, when initially writing an operator, it is recommended to use a shorter unroll_list, such as [64, 16, 4].
- When using dynamic tiering, note that different tiers may require different TileShape configurations.
- In multi-level nested loop scenarios, only the innermost loop_unroll can successfully use the unroll_list parameter.

The following is a reference example:

```python

'''
input: A, shape:[-1, 64]
output: B, shape:[-1, 64]

-1: indicates a dynamic shape
'''
.....
for b, k in pypto.loop_unroll(A.shape[0] // 64, unroll_list=[64, 16, 4], name="A", idx_name='b'):
   ### Support setting different tuning parameters at different unroll tiers
   if k <= 16:
      pypto.set_vec_tile_shapes(16, 64)
   else :
      pypto.set_vec_tile_shapes(64, 64)

   tile_a = A[b * 64:(b + k) * 64, :]
   tile_a = tile_a + 2
   B[b * 64:, :] = tile_a
```

For the tuning effect, see section 3.3.3 of [GDR Operator Cases](./performance_case_GDR.md).

### Setting Reasonable Initial TileShape Values

For the basic principles and usage constraints of TileShape configuration, see [Tiling Configuration](../development/tiling.md). On one hand, the split size directly determines the number of tasks after operator splitting, which in turn determines the number of cores used and the number of computation rounds during actual execution. On the other hand, the split size theoretically determines the arithmetic intensity of the operator. Therefore, the key to optimizing performance is optimizing the Tiling configuration.

Generally, a larger split leads to higher arithmetic intensity, making it easier for computation to reach Compute Bound and thereby fully utilizing the NPU's compute power. This is because splitting inevitably introduces repeated data transfer, and more splits result in more repeated data transfer, thus lowering the arithmetic intensity. On the other hand, the split size cannot be increased indefinitely due to the limitations of on-chip multi-level cache space (L1, L0, or UB).

#### Initial Tiling Configuration for Matmul

For cube computation scenarios, taking the case where both A and B cubes are of DT_BF16 or DT_FP16 type as an example, the recommended Tiling configurations that satisfy buffer space constraints are:

```python
# For Cube-related computation, the following TileShape is recommended. Select the closest configuration based on the actual M, K, and N sizes:
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

Advantages of the above Tiling configuration:

- It achieves high arithmetic intensity while satisfying the L0 buffer constraint. Since the tile size must meet the alignment requirements of the fractal format and the impact of the split size on read and write bandwidth must also be considered, a combination of 128–256 is generally used.
- When further in-depth tuning is performed using graph stitching-related APIs, there is an opportunity to enable Double Buffer, enabling pipeline parallelism.

#### Vector Initial Tiling Configuration

For vector computation scenarios, determine the appropriate TileShape based on the Operation and chip UB size.

- First, the TileShape must meet the specification constraints of the specific Operation. For example, scatter update requires that the TileShape of the tail axis be consistent with the Shape, meaning the tail axis is not split. For the specific constraints of each Operation, refer to the relevant API documentation.

- Second, ensure that the input and output tensors of the Operation can be allocated memory in the UB, so the TileShape must not be too large. At the same time, since small data blocks in subgraphs and data transfer can cause performance degradation, the TileShape must not be too small either. Taking Atlas A3 products as an example, the UB cache capacity is 192 KB. Therefore, a suitable initial TileShape should both meet the Operation requirements and keep the data block size between 16 KB and 64 KB, with the tail axis aligned to 32B.

- In addition, for reduction operations (Reduce operations, such as sum, max, and min), avoid splitting along the reduction axis whenever possible. For example, for RMSNorm with an input Shape of (56, 1024), its last-dimension TileShape should be set to 1024. The upper half of the figure below shows an example swimlane diagram of RMSNorm with the reduce axis split, where the outputs of multiple subgraphs need to undergo a reduce operation in the same subgraph, resulting in GM data transfer and scheduling overhead. The lower half shows an example without splitting the reduce axis, where the upstream and downstream subgraphs are stitched, eliminating GM data transfer and scheduling overhead.

![Example of reduce axis split](../../../figures/tensor/perf_reduce.png)

Based on the above considerations, you can use the following settings during the initial operator development phase, and then perform further tuning based on swimlane diagram data.

```python
# For Vector-related computation, the following TileShape is recommended:
pypto.set_vec_tile_shapes(64, 512)
```

For the actual effect of TileShape tuning, see section 3.2 of [GDR Operator Cases](./performance_case_GDR.md).

It should be noted that the above Tiling configuration is not fixed. You need to consider it comprehensively based on the computation scenario (including input Shape, Dtype, Format, etc.) and the hardware platform.

### Other Considerations

- Check whether input cubes, especially weight cubes with large shapes, can be stored in NZ format in advance. Data transfer to L1 in NZ format has higher bandwidth.

- When there is a transpose before or after cube multiplication, try swapping the left and right cubes and using the left/right cube transpose configuration. Since the N axis is on the tail axis, when the M axis is large and the N axis is small, you can also try this method to give the left and right cubes a larger tail axis, thereby improving transfer bandwidth.

- Check whether there is redundant data transfer caused by improper data operations, for example, replacing concat with assemble, or trying to configure the `inplace = True` parameter for reshape.

## Deep Performance Tuning

Further optimization of operator performance requires a man-in-loop approach: By obtaining and analyzing current operator performance data, you can make targeted adjustments to various performance configuration parameters, and gradually approach optimal performance through iterative tuning. Operator performance data can be obtained through the swimlane diagram. The collection and analysis of swimlane diagrams are an important part of the operator tuning process, and the tuning process in this chapter needs to be carried out in conjunction with swimlane diagrams.

### Stitch Tuning

The [Stitch](../../../appendix/glossary.md) configuration determines how many root functions are dispatched simultaneously, that is, this parameter controls the maximum number of loops that can be processed in a single stitch. It affects scheduling overhead, control flow generation latency, and workspace memory usage. Therefore, a larger Stitch value allows tasks to be fully parallelized, typically resulting in better performance. When a large number of gaps appear in the swimlane diagram, the Stitch value may be too small.

The current Stitch configuration is primarily determined by the [stitch_function_max_num](../../../../api/tensor_api/config/pypto-frontend-jit.md#runtime_options_detail) parameter, which is configured in the jit decorator. See the following configuration example:

```python
    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128}
    )
```

Using the glm_attention.py operator as an example, compare the impact of this value:

- When this value is too small (for example, set to 1), each task requires synchronization, resulting in high scheduling overhead and poor performance. The operator kernel latency is 1230 us, and the end-to-end latency (scheduling latency + execution latency) is 1590 us. The corresponding swimlane diagram is shown in the figure below:

![Swimlane diagram - Stitch - 1](../../../figures/tensor/stitchnum11.png)
![Trace diagram - Stitch - 1](../../../figures/tensor/stitchnum12.png)

- When this value is increased to 128, the swimlane diagram is noticeably more compact, and scheduling and synchronization overhead is significantly reduced. The operator kernel latency is 180 us, and the end-to-end latency is 803 us. The corresponding swimlane diagram is as follows:

![Swimlane diagram - Stitch - 1](../../../figures/tensor/stitchnum21.png)
![Trace diagram - Stitch - 1](../../../figures/tensor/stitchnum22.png)

- When this value is further increased to 512, the swimlane diagram becomes even more compact, but the scheduling latency increases noticeably. The operator kernel latency is 150 us, and the end-to-end latency is 977 us. The corresponding swimlane diagram is as follows:

![Swimlane diagram - Stitch - 1](../../../figures/tensor/stitchnum31.png)
![Trace diagram - Stitch - 1](../../../figures/tensor/stitchnum32.png)

As can be seen, as the Stitch value increases, the operator kernel latency continues to decrease. However, a larger Stitch configuration is not always better:

- Excessively large parameters can significantly increase scheduling overhead, making the end-to-end latency gain not worth the cost.

- When Stitch is set to a large value, the workspace also increases, and massive task parallelism may lead to a low L2 hit rate.

Tuning suggestion: If memory resources permit, you can gradually increase the Stitch configuration, and adjust the `stitch_function_max_num` parameter based on the swimlane diagram and end-to-end total latency data to find the optimal balance between performance gains and control flow overhead, thereby reducing the total latency.

For actual results, see section 3.1 in [GDR Operator Cases](./performance_case_GDR.md).

### TileShape Tuning

#### Matmul TileShape Tuning

To further tune the TileShape of Matmul, you need to fully consider its impact on arithmetic intensity and bandwidth. For details, see the [Matmul High-Performance Programming](./matmul_performance_guide.md) chapter.

This phase focuses on two tuning techniques: **reducing repeated loads** and **K-axis split**, which correspond to the `enable_split_k` configuration parameter of the `set_cube_tile_shapes` API. You can derive and select an appropriate switch configuration strategy based on the principles described above, or directly test and verify using swimlane diagram data to select the optimal configuration. The two parameters are decoupled from each other. For the specific syntax, refer to the following configuration:

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
```

#### Vector TileShape Tuning

In addition to following the principles described in the previous sections, Vector TileShape configuration must also consider the following points in conjunction with the swimlane diagram:

- For downstream Vector Operations, use the output TileShape of the upstream Operation whenever possible. For example, when a Transpose is followed by an Add Operation, if the former's TileShape is set to (64, 128), the latter's TileShape should preferably be (128, 64). When the TileShapes of upstream and downstream Operations are aligned, they have a simple one-to-one dependency, and the pass usually automatically stitches them into a single subgraph, achieving the optimization effect of a fused operator. If they are not stitched, you can use the sg_set_scope or graph splitting knob described earlier to stitch them. When the TileShapes of upstream and downstream Operations are not aligned, many-to-many dependencies may arise, preventing normal graph stitching. As shown in the figure below, each color represents a subgraph. When the preceding Sqrt and subsequent Cast Operations use the same TileShape, two parallel stitching subgraphs can be split out. Conversely, when Sqrt and Cast Operations use different TileShapes, the dependency between upstream and downstream subgraphs is three-to-two, and parallel stitched subgraphs cannot be obtained in this case.

![alt text](../../../figures/tensor/perf_tilesize.png)

- Adjust the TileShape based on the subgraph size and parallelism shown in the swimlane diagram. In the target optimization scenario, when a certain part of the swimlane diagram has a low number of parallel cores (for example, using less than half of the Vector cores), try reducing the TileShape of the Operations in that area. Conversely, when a certain part of the swimlane diagram has short subgraph latency and a high proportion of scheduling overhead, try increasing the TileShape of the Operations in that area. Note that these adjustments should avoid the tail axis and reduction axis, because the TileShape on these axes should follow the optimization principles described earlier. In addition, TileShape optimization may cause OoO errors due to graph stitching or other reasons. In such cases, identify the Operations related to the error, reduce their TileShape, and then perform the third optimization described above.

- Adjust the TileShape of adjacent Cube and Vector Operations to simplify the dependencies between Cube subgraphs and Vector subgraphs, and avoid many-to-many dependencies as much as possible.

### Graph Stitching Tuning

Before optimizing knobs related to graph stitching, you need to complete TileShape tuning first, because inappropriate TileShape can lead to complex dependencies, making it theoretically impossible to obtain subgraph tasks that are both multi-core parallel and stitched with multiple Operations.

Graph stitching refers to the process of combining multiple logically independent Operations in a compute graph into a single logical subgraph, which ultimately generates a physical compute kernel. The compute graph of a deep learning model often consists of a large number of fine-grained Operations. In the traditional per-Operation execution mode, each Operation independently triggers a kernel launch, and intermediate results are written back to global memory (GM) after computation. This execution approach introduces significant kernel launch overhead and redundant memory access on actual hardware, making it difficult to fully utilize the computing power of compute units. Graph stitching optimization enables multiple Operations to execute collaboratively within the same kernel through logical aggregation of Operations. Intermediate computation results can be retained in on-chip cache for direct reading by downstream Operations, thereby eliminating redundant GM reads and writes, significantly improving the compute-to-memory-access ratio and overall execution efficiency.

In the PyPTO programming model, developers build compute graphs using tensors and tensor Operations. The graph stitching process is automatically completed by the compiler's internal optimization pass, eliminating the need to manually write fused Operation code. The graph stitching pass analyzes and rewrites the compute graph while ensuring computational correctness, partitioning and reorganizing the original compute graph into subgraphs better suited for execution on the target hardware. PyPTO's graph stitching optimization is primarily divided into two categories: depth-wise graph stitching and breadth-wise graph stitching, targeting different performance bottleneck scenarios respectively.

#### Depth-Wise Graph Stitching

Depth-wise graph stitching is based on the producer-consumer relationship in the compute graph, fusing adjacent Operations along the data dependency path. This approach eliminates write-back operations of intermediate results and directly optimizes the data flow path, allowing operator chains originally constrained by bandwidth to complete computation seamlessly within a single kernel.
![](../../../figures/tensor/pypto.set_pass_options_1.png)

The PyPTO framework has already implemented automatic graph stitching in the depth direction. In extreme performance optimization scenarios, you can manually specify the graph stitching scheme to assign operations to specific compute tasks, thereby adjusting the latency of each task to achieve load balancing. This is done by configuring the **sg_set_scope** parameter of the [set_pass_options](../../../../api/tensor_api/config/pypto-set_pass_options.md) API.

Stitching targets typically come from dependencies between Operations. For example, when the amount of data transferred between two upstream and downstream Operations is large, they should be stitched to reduce data transfer latency. Alternatively, when multiple Operations become multiple parallel connected branches after tile splitting, these Operations should be stitched. Stitching targets can also come from inherent experience with specific operator types. For example, after splitting the batch axis, s2 axis, and g axis of an IFA operator, V1 and V2 should generally each serve as a separate subgraph task.

Currently, this capability is mainly considered for use in continuous Vector computation processes, and graph stitching between Matmul Operations and Vector Operations is not yet supported. When using it, you need to analyze and adjust based on the swimlane diagram information.

#### Breadth-Wise Graph Stitching

Breadth-wise graph stitching targets Operations that are at the same level in the computation graph and can be executed in parallel. It merges multiple parallel Operations into the same Kernel for execution to increase the computation scale of a single Kernel. During the intra-core instruction scheduling phase, multi-branch stitching can more fully fill the hardware pipeline and achieve better multi-pipe concurrency. At the memory access level, by merging accesses from the same source, multiple repeated GM-to-L1 transfers are consolidated into a single load, which saves memory bandwidth while effectively amortizing the Kernel launch overhead, ultimately improving the overall throughput of the hardware compute units.
![](../../../figures/tensor/pypto.set_pass_options_2.png)

For Matmul and Vector computation, PyPTO provides different breadth-wise graph stitching tuning APIs, which will be introduced in detail in the following sections.

**Matmul Breadth-Wise Graph Stitching**

In Matmul computation scenarios, configure subgraph stitching through the [set_pass_options](../../../../api/tensor_api/config/pypto-set_pass_options.md) API. You can mainly choose between two strategies: `L1Reuse` and `CubeNBuffer`. Both are used to merge Cube subgraphs in the breadth direction. The former is used to merge subgraphs with redundant L1 transfers, and the latter is used to merge isomorphic subgraphs. L1Reuse can reduce the L1 transfer volume, and CubeNBuffer can hide the transfer and computation latency between cube multiplications of different branches. Both can reduce subgraph scheduling overhead. However, considering that the performance bottleneck in most cube multiplication scenarios is data transfer, prioritize the L1Reuse strategy to reduce the data transfer volume.

In fact, the L1Reuse strategy is enabled by default, and the number of subgraphs to stitch is automatically calculated and configured. For extreme performance optimization scenarios, you can manually configure it via the `cube_l1_reuse_setting` parameter, typically considering values such as 2, 4, or 8, and select the optimal value based on measured data from the swimlane diagram.

`cube_l1_reuse_setting` supports function-level configuration (in the `func{magic}_{order}` format), enabling fine-grained control across different root functions: it affects only the specified function without impacting others. The configuration information is directly displayed in the hashOrder-hint field of the swimlane diagram (in the format `l1ReuseInfo hashOrder: func8_0, subGraphCount: 24`), allowing you to match the stitch granularity based on the subgraph count (subGraphCount) and the number of cores. It also supports configuration based on semantic_label, as shown in the following example:

```python
# Function-level configuration
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {"DEFAULT": 4, "func8_0": 1, "func8_1": 1}}
)

# semantic_label configuration (can coexist with function-level keys)
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {"C1": 4}}
)
```

CubeNBuffer targets scenarios where L1Reuse cannot be enabled. Such scenarios are relatively rare and mainly include the following two cases:

1. There is no repeated L1 data transfer between Cube subgraphs. For example, when the left and right cube shapes of BatchMatmul are (128, 64, 64) and (128, 64, 64) respectively, after the pass splits out 128 isomorphic Cube subgraphs with left and right cube shapes of (64, 64) and (64, 64) respectively, there is no repeated L1 data transfer between them. Similarly, for the MM2 of the FA operator, there is no repeated L1 data transfer between MM2 subgraphs of different S2 blocks.
2. The K axis is long. When K-axis split is not performed, L1Reuse requires an entire row of the left cube or an entire column of the right cube to reside in L1. However, the L1 cache capacity is limited. Therefore, when the K axis is long and K-axis split is not performed, L1Reuse cannot be used.

In this case, you can configure the `cube_nbuffer_setting` parameter and further tune it based on the measured data in the swimlane diagram.

**Vector Breadth-Wise Graph Stitching**

In vector computation scenarios, configure the breadth-wise graph stitching operation via the `vec_nbuffer_setting` parameter of the [set_pass_options](../../../../api/tensor_api/config/pypto-set_pass_options.md) API. Note that you should first perform the preceding optimization steps to adjust the splitting and merging of upstream and downstream subgraphs to a suitable state before attempting to use vecNBuffer for breadth-wise merging. When the swimlane diagram shows isomorphic subgraph groups with a large number of small subgraphs (with latency below 10us), you should use this feature for optimization to reduce scheduling overhead and kernel header overhead.

The configuration method of the `vec_nbuffer_setting` parameter is similar to that of `cube_nbuffer_setting`.

### Scheduling Strategy Tuning

The inter-core pipelining of a PyPTO operator is determined by the AI CPU's scheduling of subgraphs, which is based on the dependencies between subgraphs and the scheduling strategy for inter-core tasks. You can try changing this scheduling strategy to achieve better operator performance. When the dependencies between upstream and downstream subgraphs are relatively simple, or when the L2 hit rate of the input tensor of the downstream subgraph is critical, L2 affinity scheduling is recommended. The configuration method is as follows:

```python
@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
```

During specific configuration, you should consider the impact of both L2 reuse and load balancing. The optimal configuration strategy varies across different scenarios and should be analyzed in conjunction with the swimlane diagram.

### Other Tuning Methods

- For special shapes, you can try using Vector operations to preprocess the input cube to make it a more standard shape. Take cube multiplication with left and right cube shapes of (884736, 16) and (16, 16) respectively as an example. If only L1Reuse is used, it can only be optimized to 500 us. However, by using the following approach — concatenating four repeated right cubes diagonally into a new right cube c with a shape of (64, 64) in advance, and then performing cube multiplication with the reshaped left cube — the operator performance is significantly improved to 40 us.

```python
def matmul_kernel(a, b, out):
    # Construct c
    pypto.set_vec_tile_shapes(64, 64)
    d = pypto.full([16, 16], 0.0, pypto.DT_BF16)
    c1 = pypto.concat([b, d, d, d], 1)
    c2 = pypto.concat([d, b, d, d], 1)
    c3 = pypto.concat([d, d, b, d], 1)
    c4 = pypto.concat([d, d, d, b], 1)
    c = pypto.concat([c1, c2, c3, c4], 0)
    # Reshape a
    a = pypto.reshape(a, [221184, 64])
    # Cube multiplication
    pypto.set_pass_options(cube_l1_reuse_setting={"DEFAULT": 9})
    pypto.set_cube_tile_shapes([512, 512], [64, 64], [64, 64], True)
    e = pypto.matmul(a, c, pypto.DT_BF16)
    e = pypto.reshape(e, [884736, 16])
    pypto.assemble(e, [0, 0], out)
```

- Add redundant computation to avoid redundant dependencies and data movement. For example, by copying e_score_bias_2d into tile_batch copies and then performing the cast operation, the cast of each copy is stitched with the other operations of the corresponding batch. This avoids the one-to-many subgraph dependencies between the cast operation of e_score_bias_2d and the subsequent computation of each batch, which would increase scheduling overhead, and also avoids the data movement of the cast result.

```python
e_score_bias_2d_tile = pypto.tensor([tile_batch, ne], e_score_bias_2d.dtype, "e_score_bias_2d_tile")
for tmp_idx in range(tile_batch):
       pypto.assemble(e_score_bias_2d, [tmp_idx, 0], e_score_bias_2d_tile)
e_score_bias_2d_cast = pypto.cast(e_score_bias_2d_tile, tile_logits_fp32.dtype)
```

- Avoid processing tensors with small tail axis lengths whenever possible. When even a larger TileShape cannot prevent the input tensor of an Operation from having a small tail axis, use data manipulation Operations such as concat, transpose, or reshape to enlarge the tail axis.

- Set an appropriate L2 CacheMode via [set_cache_policy](../../../../api/tensor_api/tensor/pypto-Tensor-set_cache_policy.md). For global memory data that is accessed only once, set its access state to not enter the L2 cache.

- If operator performance remains poor after the preceding tuning measures, consider whether the TileOperation implementation itself is suboptimal. You can construct a standalone Operation case and compare its performance with that of an Ascend C small operator. If the performance is confirmed to be poor, check whether more optimal instructions could be used.
