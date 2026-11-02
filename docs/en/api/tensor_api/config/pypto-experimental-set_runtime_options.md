# pypto.experimental.set\_runtime\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:20:57.867Z pushedAt=2026-10-28T11:57:29.538Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->

## Description

This API is used to set **experimental runtime configurations**. It converts the parameters in `tile_fwk_config.json` that are not yet stable in the runtime into programmable APIs. Subsequent experimental runtime features are also extended through this API and are not added to the stable parameter list of `pypto.frontend.jit(runtime_options=...)`.

## Prototype

```python
set_runtime_options(
    *,
    stitch_function_num_per_pool: Optional[list[int]] = None,
)
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| stitch_function_num_per_pool | Input | Meaning: Used to separately control the capacity of the three sub-memory pools in Workspace, in the format `[root_inner, assemble_outcast, exclusive_outcast]`. The configured size determines the maximum number of stitch build loop iterations that each memory pool can accommodate.<br> Description: Based on the actual usage of each sub-memory pool during operator runtime, finely control the reserved capacity separately. This configuration can reduce memory usage while meeting the target performance, or improve performance when memory usage is prioritized.<br> Configuration method: `[0, 0, 0]` means disabled; if any element is non-zero, it is enabled. After enabling, a dimension set to 0 means no capacity is reserved for that sub-memory pool. If the operator has intermediate tensors of the type corresponding to that memory pool, compilation fails due to insufficient capacity, and that dimension must be set to at least 1; it can be set to 0 only when the operator has no tensors of that type. For the meaning of the three sub-memory pools and their corresponding tensor types, see [Example 1](#stitch_function_num_per_pool_detail).<br> Usage scenarios:<br> (1) When parallelism already meets expectations and fine-grained memory control is required to reduce memory usage while maintaining performance: first use `stitch_function_max_num` to tune parallelism to the target performance, then lower the three sub-memory pools separately according to the recommended values to reduce memory usage.<br> (2) In memory-priority scenarios, when you expect to improve performance through fine-grained memory control: first use `max_workspace_kb` to control the total memory, then finely control the size of the sub-memory pool that actually limits the number of stitch build loop iterations to improve performance.<br> For detailed tuning steps, see [Tuning Method](#workspace_tuning_method).<br> Type: list of int, fixed 3 elements<br> Value range: each element 0 to 1024<br> Default value: `[0, 0, 0]`<br> Affected pass scope: NA |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

- Type safety: Must be a list or tuple of length 3, with elements being integers in the range `[0, 1024]`; bool is not allowed.
- Scope: Do not set it inside `pypto.loop` or inside a kernel.
- Relationship between configuration items: `stitch_function_max_num` is used when memory usage is not considered and only performance tuning is considered; `max_workspace_kb` is used to compress the total memory when there is a memory limit, which may reduce parallelism. Fine-grained configuration is used after the above two items, setting the three sub-memory pools separately. After enabling, both `stitch_function_max_num` and `max_workspace_kb` become invalid.

## Examples

```python
pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[64, 1, 1])
```

### Example 1: Fine-grained Workspace Mode <a id="stitch_function_num_per_pool_detail"></a>

The following example illustrates the difference between the default and fine-grained control:

```python
B_STATIC, L_STATIC, H_STATIC, D_STATIC = 1, 64, 1, 16

pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[64, 1, 1])

@pypto.frontend.jit
def k_tmp_to_d_emb(
    dy: pypto.Tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    output1: pypto.Tensor([B_STATIC, L_STATIC, D_STATIC], pypto.DT_FP32),
    output2: pypto.Tensor([B_STATIC, L_STATIC, D_STATIC], pypto.DT_FP32),
):
    tmp_assemble = pypto.tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], output1.dtype, "tmp_assemble")
    tmp_exclusive = pypto.tensor([B_STATIC, L_STATIC, H_STATIC, D_STATIC], output2.dtype, "tmp_exclusive")

    # Loop0: Exclusive write
    for i_idx, t in pypto.loop_unroll(0, 1, 1, name="l_loop_0"):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        tmp_exclusive[:] = pypto.add(dy, dy)

    # Loop1: Assemble write
    for j_idx, t in pypto.loop_unroll(0, L_STATIC, 1, name="l_loop_1"):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        dy_v = dy[0, j_idx : j_idx + t, 0]
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        dx = pypto.matmul(dy_v, weight[0], pypto.DT_FP32, b_trans=True)
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        tmp_assemble[0, j_idx : j_idx + t, 0] = dx + 0.0

    # Loop2: Read
    for k_idx, t in pypto.loop_unroll(0, L_STATIC, 1, name="l_loop_2"):
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        output1[0, k_idx : k_idx + t] = tmp_assemble[0, k_idx : k_idx + t, 0]
        output2[0, k_idx : k_idx + t] = tmp_exclusive[0, k_idx : k_idx + t, 0]
```

The tensor memory distribution in this example is as follows:

| Tensor | Data Ownership |
| --- | --- |
| dy, weight, output1, output2 | Input parameters, not placed in the Workspace memory pool |
| dy_v | Slice view of dy, reusing the same memory |
| dx | RootInner |
| tmp_assemble | Assemble outcast |
| tmp_exclusive | Exclusive outcast |

After `stitch_function_num_per_pool` is configured to a non-all-zero value, the numbers of loops built by stitch that the three sub-memory pools can accommodate become independent of each other and can be set separately according to actual requirements. For this example, the recommended configuration derived by the runtime from the actual occupancy is `[64, 1, 1]`, as described below:

- **root_inner**: Reduced from the default 128 to 64. `dx` exists only within a single iteration of loop1, so no capacity needs to be reserved for loop0 or loop2.
- **assemble_outcast**: Reduced from the default 128 to 1. `tmp_assemble` is created outside the loop and shares the same memory across multiple iterations.
- **exclusive_outcast**: Lowered from the default 128 to 1. `tmp_exclusive` is generated only in one iteration of loop0.

With the preceding settings, the workspace usage is reduced while the parallelism remains essentially unchanged.

## Tuning Method <a id="workspace_tuning_method"></a>


### Collecting Logs

```bash
export ASCEND_PROCESS_LOG_PATH=./wk
export ASCEND_GLOBAL_LOG_LEVEL=1
python your_test.py
```

```bash
grep -rE "\[Workspace Runtime (Pool|Tuning|Summary|Recommendation)\]" ./wk
```

| Log Identifier | Description | Key Fields |
|----------|------|----------|
| `[Workspace Runtime Pool]` | Actual usage of each pool when a single task is submitted | `current`, `peak`, `capacity` |
| `[Workspace Runtime Tuning]` | Configuration and recommendation for a single task submission | `taskId`, `stitchCount`, `configuredDepths`, `recommendedDepths` |
| `[Workspace Runtime Summary]` | Maximum actual usage and capacity limit of each pool across the entire execution | `peak` and `capacity` of each pool |
| `[Workspace Runtime Recommendation]` | Configuration recommendation after the entire execution, and the memory increase when a pool's configuration value is incremented by 1 | `recommendedDepths`, `rawPerParallelBytes`, `nextDepthTotalBytes` |

`recommendedDepths={rootInner, assembleOutcast, exclusive}` corresponds to the three elements of `stitch_function_num_per_pool` in order. `actualDepths` is the current effective size of each memory pool.

### Scenario 1: Parallelism Meets Expectations, Fine-Tune to Reduce Memory Usage

Target: When parallelism has reached the expected level, remove memory that have been allocated but are not in use to reduce the workspace while maintaining performance.

Procedure:

1. Use only `stitch_function_max_num` (or leave it unset, defaulting to 128) to tune end-to-end performance to the target. Do not enable this configuration or `max_workspace_kb` at the same time.
2. Enable INFO logging as described above and execute the operator. If `[Workspace Runtime Alloc Failure]` appears, parallelism has been truncated by memory. You must first increase `stitch_function_max_num` or switch to Scenario 2.
3. Read `recommendedDepths` from `[Workspace Runtime Recommendation]` and write it into this configuration. For example, `recommendedDepths={rootInner=64, assembleOutcast=1, exclusive=1}` corresponds to `[64, 1, 1]`.
4. Remove the `stitch_function_max_num` configuration and retest: the workspace should decrease, and the end-to-end latency should remain roughly the same as in step 1. If performance degrades, check `[Workspace Runtime Summary]` for the memory pool whose actual usage is close to its capacity limit, increment that element by 1, and retry.
5. Cover the shapes and execution paths actually used by the operator, take the maximum value of each item in `recommendedDepths` across all samples, finalize the configuration, and then disable INFO logging for acceptance.


### Scenario 2: Prioritize memory, refine control to improve parallelism

Target: When the total memory usage is limited, finely control the sub-memory pool sizes so that memory is allocated to the pool that actually limits the number of loops for stitch construction.

`max_workspace_kb` compresses the three sub-memory pools at the same scale. The recommended value reflects the parallelism derived from the actual memory usage under the current memory limit, not the optimal parallelism when memory is sufficient, and it does not guarantee that memory is fully utilized. When memory is not fully used, you can improve parallelism by following the steps below.

Procedure:

1. Use `max_workspace_kb` (which must be greater than the minimum runnable value indicated in the prompt). For example, when the limit is 200 MB, configure `max_workspace_kb=200*1024`.
2. Collect logs as described above. From `[Workspace Runtime Summary]`, compare the actual usage and capacity limit of each pool. From `[Workspace Runtime Recommendation]`, read `recommendedDepths` and `nextDepthTotalBytes`.
3. Write `recommendedDepths` into this configuration as the starting point, and remove `max_workspace_kb`.
4. Calculate the remaining memory: the limit minus the current workspace usage. Increasing the configuration value of a memory pool by 1 increases the bytes for the entire execution by the corresponding item in `nextDepthTotalBytes`.
5. Prioritize increasing the pool whose actual usage is closer to its capacity limit and whose `nextDepthTotalBytes` can be covered by the remaining memory (commonly the first element, `root_inner`). Increase only that element by 1 each time, and retest the end-to-end latency and workspace.
6. Stop when increasing by 1 more would exceed the limit, or when the latency no longer decreases. For multiple shapes, take the maximum value of each element and fix it.

Example: With a quota of 200 MB and an actual peak of about 100 MB, `recommendedDepths={rootInner=32, assembleOutcast=1, exclusive=1}`. The remaining approximately 100 MB can be used to increase `root_inner`, and the number of increments must not exceed the remaining bytes divided by `nextDepthTotalBytes.rootInner`. After the increase, the workspace grows and the parallelism increases. If `[Workspace Runtime Alloc Failure]` appears or performance regresses, decrease this element back.
