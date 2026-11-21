# pypto.experimental.set\_runtime\_options

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Sets the **experimental runtime configuration**. It converts the parameters that are not yet stable in the runtime of `tile_fwk_config.json` into programmable APIs. The runtime experimental features added later are also extended through this API and are not included in the stable parameter table of `pypto.frontend.jit(runtime_options=...)`.

## Prototype

```python
set_runtime_options(
    *,
    stitch_function_num_per_pool: Optional[list[int]] = None,
)
```

## Parameters

| Parameter| Input/Output| Description|
|--------|-----------|------|
| stitch_function_num_per_pool | Input| Meaning: Controls the capacity of the three sub-memory pools in the workspace, in the format `[root_inner, assemble_outcast, exclusive_outcast]`. The configured size determines the maximum number of loop iterations that can be built by Stitch in each memory pool.<br> Description: The reserved capacity is finely controlled based on the actual usage of each sub-memory pool during operator execution. This configuration can be used to reduce the memory usage while meeting the target performance, or improve the performance when the memory is preferentially used.<br> Configuration method: `[0, 0, 0]` indicates that this option is disabled. If any element is not 0, this option is enabled. Once enabled, if the value of a dimension is 0, it indicates that no capacity is reserved for the sub-memory pool. If the operator has an intermediate tensor of the type corresponding to the memory pool, the compilation will fail due to insufficient capacity. In this case, you need to set the dimension to at least 1. The value 0 can be set only when the operator does not have a tensor of this type. For details about the meanings of the three sub-memory pools and the corresponding tensor types, see [Example 1](#stitch_function_num_per_pool_detail).<br> Scenarios:<br> (1) When parallelism already meets expectations and fine-grained memory control is required to reduce memory usage while maintaining performance: You can use `stitch_function_max_num` to tune parallelism to the target performance, and then reduce the size of the three sub-memory pools separately based on the recommended values to reduce the memory usage.<br> (2) In the scenario where the memory is preferentially used, fine-grained memory control is required to improve the performance: You can use `max_workspace_kb` to control the total memory, and then finely control the size of the sub-memory pool that limits the number of loop iterations built by Stitch to improve the performance.<br> For detailed tuning steps, see [Tuning Method](#workspace_tuning_method).<br> Type: list of int. Three elements are fixed.<br> Value range: 0 to 1024 for each element<br> Default value: `[0, 0, 0]`<br> Affected Pass scope: N/A|

## Return Value

void: The Set method does not return any value. The setting takes effect immediately after being successfully set.

## Constraints

- Type safety: The value must be a list or tuple with three elements, and the elements must be integers in `[0, 1024]`. Boolean values are not allowed.
- Scope: Do not set this parameter in `pypto.loop` or kernel.
- Relative relationship between configuration items: `stitch_function_max_num` is used when the memory usage is not considered and only the tuning performance is considered. `max_workspace_kb` is used when the memory is compressed based on the total memory size when the memory is limited, which may reduce parallelism. Fine-grained configuration is used after the preceding two items. It is configured separately for the three sub-memory pools. Once enabled, both `stitch_function_max_num` and `max_workspace_kb` become invalid.

## Examples

```python
pypto.experimental.set_runtime_options(stitch_function_num_per_pool=[64, 1, 1])
```

### Example 1: Fine-grained Workspace Mode <a id="stitch_function_num_per_pool_detail"></a>

The following example describes the differences between the default and fine-grained control modes:

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

The tensor memory distribution in this example is shown in the table below.

| Tensor | Data Ownership|
| --- | --- |
| dy, weight, output1, output2| Input parameter, which is not stored in the workspace memory pool.|
| dy_v | Slice view of dy, which reuses the same memory block.|
| dx | RootInner |
| tmp_assemble | Assemble outcast |
| tmp_exclusive | Exclusive outcast |

After `stitch_function_num_per_pool` is set to a non-zero value, the number of loops built by Stitch for the three sub-memory pools is independent of each other. You can set the number of loops for each sub-memory pool based on the actual requirements. In this example, the recommended configuration is `[64, 1, 1]` based on the actual usage. The details are as follows:

- **root_inner**: The value is reduced from 128 to 64 by default. `dx` exists only in a single loop of loop1. Therefore, you do not need to reserve capacity for loop0 and loop2.
- **assemble_outcast**: The value is reduced from 128 to 1 by default. `tmp_assemble` is created outside the loop and shared by multiple loops.
- **exclusive_outcast**: The value is reduced from 128 to 1 by default. `tmp_exclusive` is generated only in one loop of loop0.

With the preceding settings, the workspace usage is reduced while the parallelism remains unchanged essentially.

## Optimization Method <a id="workspace_tuning_method"></a>


### Log Collection

```bash
export ASCEND_PROCESS_LOG_PATH=./wk
export ASCEND_GLOBAL_LOG_LEVEL=1
python your_test.py
```

```bash
grep -rE "\[Workspace Runtime (Pool|Tuning|Summary|Recommendation)\]" ./wk
```

| Log Identifier| Description| Key Fields|
|----------|------|----------|
| `[Workspace Runtime Pool]` | Actual usage of each pool when a single task is submitted| `current`, `peak`, `capacity`|
| `[Workspace Runtime Tuning]` | Configuration and recommendation for a single task submission| `taskId`, `stitchCount`, `configuredDepths`, `recommendedDepths`|
| `[Workspace Runtime Summary]` | Maximum actual usage and capacity limit of each pool during the entire execution| `peak` and `capacity` of each pool|
| `[Workspace Runtime Recommendation]` | Recommended configuration after the entire execution is complete, and the memory increases when a pool's configuration value is incremented by 1| `recommendedDepths`, `rawPerParallelBytes`, `nextDepthTotalBytes`|

`recommendedDepths={rootInner, assembleOutcast, exclusive}` corresponds to the three elements of `stitch_function_num_per_pool` in sequence. `actualDepths` indicates the current effective size of each memory pool.

### Scenario 1: Fine-Tuning to Reduce Memory Usage When Parallelism Meets the Expectation

Objective: When parallelism meets the expectation, the memory that has been allocated but not used is released to reduce the workspace while maintaining the performance.

Procedure:

1. Use only `stitch_function_max_num` (or do not set it, which defaults to 128) to tune the end-to-end performance to the target. Leave both this configuration and `max_workspace_kb` disabled
2. Enable the INFO logging function and execute the operator. If `[Workspace Runtime Alloc Failure]` is displayed, the parallelism has been truncated by the memory. In this case, increase the value of `stitch_function_max_num` or go to scenario 2.
3. Read `recommendedDepths` from `[Workspace Runtime Recommendation]` and write it to this configuration. For example, `recommendedDepths={rootInner=64, assembleOutcast=1, exclusive=1}` corresponds to `[64, 1, 1]`.
4. Remove the `stitch_function_max_num` configuration and retest. The workspace should decrease, and the end-to-end latency should be basically the same as that in step 1. If the performance deteriorates, check `[Workspace Runtime Summary]` for the memory pool whose actual usage is close to the capacity limit. Increment that element by 1, and retry.
5. Cover the shapes and execution paths actually used by the operator, take the maximum value of each item in `recommendedDepths` across all samples, finalize the configuration, and then disable INFO logging for acceptance.


### Scenario 2: Fine-grained Control to Improve Parallelism When Memory Is Preferentially Used

Objective: When the total memory usage is limited, precisely control the size of each sub-memory pool so that memory is allocated to limit the number of loops built by Stitch.

`max_workspace_kb` compresses three sub-memory pools of the same scale. The recommended value reflects the parallelism inferred based on the actual used memory under the current memory limit, not the optimal parallelism when memory is sufficient. Therefore, it cannot ensure that memory is fully utilized. If memory is not fully used, you can take the following steps to improve the parallelism.

Procedure:

1. Use `max_workspace_kb` (which must be greater than the minimum runnable value indicated in the prompt). For example, if the limit is 200 MB, set `max_workspace_kb` to `200*1024`.
2. Collect logs as described above. Compare the actual usage and limit of each pool in `[Workspace Runtime Summary]`, and read `recommendedDepths` and `nextDepthTotalBytes` from `[Workspace Runtime Recommendation]`.
3. Write `recommendedDepths` to this configuration as the start point, and cancel `max_workspace_kb`.
4. Compute the remaining memory: the limit minus the current workspace usage. Increasing the configuration value of a memory pool by 1 increases the bytes for the entire execution by the corresponding item in `nextDepthTotalBytes`.
5. Preferentially increase the pool whose actual usage is closer to the capacity limit and whose `nextDepthTotalBytes` can be covered by the remaining memory. (The first element `root_inner` is commonly used.) Increase the value of the element by 1 each time, and test the end-to-end latency and workspace again.
6. Stop the operation when the value is about to exceed the limit or the latency does not decrease. If there are multiple shapes, the maximum value of each element is used and then fixed.

Example: The limit is 200 MB, the actual peak value is about 100 MB, and the `recommendedDepths={rootInner=32, assembleOutcast=1, exclusive=1}`. The remaining approximately 100 MB can be used to increase `root_inner`. The number of increments must not exceed the remaining bytes divided by `nextDepthTotalBytes.rootInner`. After the increase, the workspace and parallelism increase. If `[Workspace Runtime Alloc Failure]` appears or performance regresses, reduce the elements back to the original values.
