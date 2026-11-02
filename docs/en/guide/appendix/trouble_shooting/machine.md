# F7XXXX-F8XXXX

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T01:25:23.076Z pushedAt=2026-10-27T03:47:11.170Z nl=2 -->

## F70001 COMPILE_AICORE_FAILED

**Error Description**

The AI Core kernel compilation phase fails.

**Possible Causes**

- CCE source code generation fails (`GEN_AICORE_FILE_FAILED`).
- CCE compilation command execution fails (`COMPILE_CCEC_FAILED`).
- Linking fails (`LINK_FAILED`), commonly due to unresolved symbols in parallel compilation.

**Solution**

1. Check the CCE compilation command and error output in the log.
2. If `ld.lld: error: undefined symbol` appears, change `"parallel_compile": 1` in `tile_fwk_config.json` to serial compilation.

## F71004 LAUNCH_AICORE_FAILED

**Error Description**

The AI Core kernel fails to be launched from the host side.

**Possible Causes**

NA

**Solution**

1. Check whether the compilation output `kernel_aicore/*.o` is generated successfully.
2. Check whether the environment variable `ASCEND_HOME_PATH` is correct.

## F71005 RANGE_VERIFY_FAILED

**Error Description**

During the encode phase, the `InitRawTensorAndMemoryRequirement` assertion fails: `Shape size mismatch` or `Data size mismatch`, accompanied by `actualRawmagic`, `rawShape`, and `actualrawShape` output.

**Possible Causes**

The rawShapeSize or rawDataSize of the two ends of the memory reuse chain (rawTensor and its actualRaw) are inconsistent. This commonly occurs when the actualRawmagic is omitted during operations such as reshape, assemble, or view.

**Solution**

1. Record `rawMagic`, `actualRawmagic`, and `rawShape` from the assertion log.
2. Enable the pass compute graph dump (`"dump_graph": true` in `tile_fwk_config.json`).
3. Under `build/output/pass/`, focus on the tilegraphs from **pass4 to pass27**, and search for the corresponding rawMagic.
4. Analyze along the data flow to determine whether a certain pass modified the rawshape on one side without synchronizing the other side.
5. **Determine ownership**:
   - The use case does not comply with API constraints (inplace, valid_shape consistency, etc.) → Adjust the use case.
   - The use case is correct → Contact the pass developer to investigate missing actualRawmagic during transfer.

## F71008 MAP_REG_ADDR_FAILED

**Error Description**

Host-side register address mapping fails: `Map reg addr fail, maybe others are using current device`.

> **Nature**: In most cases, this is a **transient contention** in device register mapping, not a hardware fault. Retrying at short intervals on the same device often recovers the issue. Do not scan for or switch to another device before diagnosis.

**Possible Causes**

- Multiple processes/workers/containers/agents concurrently map the same `DEVICE_ID`, causing mapping window conflicts (the most common cause).
- A residual process on the same device has not released the `/dev/davinci*` context.

**Solution**

1. **Retry in place first (preferred)**
   Keep `TILE_FWK_DEVICE_ID` (or `ASCEND_DEVICE_ID`) unchanged and rerun the same test case at short intervals (2 to 3 times recommended). Most F71008 errors disappear after retries.
2. **Full-scale diagnosis (only if the failure persists)**
   - Run `npu-smi info` to check the overall occupancy.
   - Run `fuser -v /dev/davinci<CHIP_ID>` to locate the occupying process. After killing it with a timeout, confirm that the residual process has been fully cleaned up.
   - An HBM occupancy of about 3 to 9 GB usually indicates that a live NPU context still exists.
   - In container or multi-process scenarios, verify that the device number of each process is set correctly and does not conflict.
3. **When constrained by the device: wait and retry on the same device**
   **Do not** have multiple agents/processes scan idle cards and switch devices for each other — concurrent card scanning will trigger cascading F71008 errors (historically this can cause multiple cards across the entire machine to hang).
4. **Only when long-term exclusive access is confirmed or a hardware exception occurs**, switch to an idle device or restart the NPU driver.

## F7100B SYNC_FAILED

**Error Description**

Host-side stream synchronization fails: `DynamicLaunchSynchronize` in `LaunchAicoreKernel` returns a non-zero value, indicating a synchronization exception between the schedule stream and the AI Core stream.

**Possible Causes**

- Stream synchronization fails after an AI Core execution exception.
- Stream synchronization fails after profiling data synchronization in debug mode.
- The underlying runtime stream synchronization times out or its status is abnormal.

**Solution**

1. Check whether `runtime_debug_mode` is enabled or `torch.npu.synchronize()` is manually called, and temporarily disable them to eliminate issues caused by synchronization timing.
2. View the device log for abnormal information (such as timeout, task abort, and stream abort) around the time of the synchronization failure.
3. In the full-network scenario, check whether there are stream dependency conflicts with other components.

## F72002 HANDSHAKE_TIMEOUT

**Error Description**

The schedule AI CPU and AI Core handshake times out, and the scheduling thread cannot start normally.

**Possible Causes**

- The NPU device is unavailable or the driver is abnormal.
- NPU resource usage in the current process or container is excessively high, with multiple processes contending for the same device.
- The handshake timeout configuration is too short and does not match the environment.

**Solution**

1. Run `npu-smi info` to confirm that the device and driver are normal.
2. Check whether the NPU usage in the process or container is excessively high.
3. Check the log context (such as after `Schedule run init succ` and content related to AbnormalStop) to distinguish between a first handshake failure and a runtime exception.
4. Use the associated skill [pypto-environment-setup](../../../../../.agents/skills/pypto-environment-setup/SKILL.md).

## F73001 CTRL_FLOW_EXEC_FAILED

**Error Description**

Ctrl AI CPU control flow execution fails (devTask construction or stitch processing exception).

**Possible Causes**

- The allocCtx or stitchCtx of the root function is empty.
- Control flow initialization fails.
- Task stats is abnormal.

**Solution**

1. Check the device log to identify the specific failed node (`DEV_TASK_BUILD` / `ROOT_STITCH`).
2. Check the Ctrl AI CPU log context to confirm whether there is a `CELL_MATCH`-related exception.
3. If accompanied by stitch dependency exceptions (precision issues or suspected missing dependency edges), enable `runtime_debug_mode=3` for runtime dependency verification:

   ```python
   @pypto.frontend.jit(debug_options={"runtime_debug_mode": 3})
   ```

   After executing the use case, run the following command in the output directory:

   ```bash
   python tools/verify_dep_correctness.py <dump_dir>
   ```

   Validation rules:

   | Rule | Content |
   |---|---|
   | `rule_static_integrity` | Whether static successors declared at compile time are retained at runtime |
   | `rule_stitch_legality` | Whether the producer/consumer referenced by a stitch edge is valid |
   | `rule_cell_write_conflict` | Whether concurrent write conflicts exist on the same cell |

   If no issues are found, `PASS` is output. If issues are found, a categorized summary and `dep_check_report.csv` are output for detailed location.

## F73008 CTRL_ALLOC_TIMEOUT

**Error Description**

Ctrl AI CPU execution times out, or AI CPU execution times out in a full-network environment.

**Possible Causes**

- In the full-network scenario, components other than PyPTO also use AI CPU, and forced same-cluster allocation leads to insufficient resources.
- `launch_sched_aicpu_num` is improperly configured.

**Solution**

1. In the full-network scenario, set `PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false` and configure the number of available AI CPUs through `launch_sched_aicpu_num`.
2. Note: `launch_sched_aicpu_num` does not take effect when same-cluster allocation is enabled.

## F7400B WORKSPACE_CAPACITY_INSUFFICIENT

**Error Description**

Memory allocation fails, with the following symptoms:

- `torch.OutOfMemoryError: NPU out of memory`
- `rtMalloc failed. size:xxx`

**Possible Causes**

- The workspace budget is overestimated (an anomaly in one of the four categories: Tensor Workspace, Metadata, AI Core Spilled, or Debug).
- An untiled oversized tensor enters a subgraph (in exceptional scenarios such as Inplace, FixedAddress, or shmemData).
- The number of boundary Outcast slots is excessive, or the size of an individual slot is abnormal.

**Solution**

1. **Locate the abnormal category**: Enable INFO logs, search for `[workspaceSize]`, and compare the items `Metadata / tensor / aicoreSpilled / debug`.
2. **Narrow down to the root function**: Each root has an independent log `MaxRootInnerMem is xxx`, and the one with the largest value is the source of the problem.
3. **Locate the problematic tensor**: Search for the `staticMemReq=[xxx] is too large` warning, and locate it in the pass compute graph based on rawmagic.
4. **Adjust the configuration**:
   - `max_workspace_kb` → **Recommended priority configuration**, which enables the memory-driven stitch mode. This is especially useful when `unroll_list` is used; set it to the recommended value in the encode log.
   - `stitch_function_max_num` → Controls the stitch depth (by the number of root functions) when the memory-driven mode is not enabled.
   - `unroll_list` / `max_unroll` → Control the number of unrolls.
5. **Related Skill**: [pypto-machine-workspace](../../../../../.agents/skills/pypto-machine-workspace/SKILL.md).

## F7400X Workspace Memory Overlap / Precision Issue

**Error Description**

The operator precision is abnormal, and workspace memory reuse overlap or corruption is suspected in MACHINE.

**Solution**

1. **Check input initialization**: Ensure that the inputs and outputs are correctly initialized.
2. **Check tensor continuity**: MACHINE-related operators require contiguous input/output (`tensor.is_contiguous()`). Non-contiguous tensors may cause exceptions.
3. **Expand workspace**: In `python/pypto/frontend/parser/entry.py`, expand `workspace_tensor` by 10 times. If the issue disappears, it indicates that the workspace size is underestimated.
4. **Internal self-management of workspace**: Modify `PrepareDevProgArgs` in `framework/src/machine/runtime/device_launcher.cpp` to disable external workspace passing and use internal `AllocDev` instead.
5. **Leaf-level memory overlap detection**: Enable `ENABLE_DUMP_OPERATION=1` + `runtime_debug_mode=1`, and after running, execute:

   ```bash
    python3 tools/schema/schema_memory_check.py -d <device_log_dir> -t <dyn_topo.txt_path>
    ```
