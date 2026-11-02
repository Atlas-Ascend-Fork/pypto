# pypto.frontend.jit

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:21:39.864Z pushedAt=2026-10-28T11:57:29.542Z nl=2 -->

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

`pypto.frontend.jit` is the core decorator in the frontend architecture. It just-in-time (JIT) compiles Python functions into efficient computation graphs and executes them on the NPU. The frontend does not support return values and only supports in-place modification. It supports passing in torch tensors and variables of other types.

Key features:

- **In-place modification**: Kernel functions pass computation results by modifying output tensors in place, and return values are not supported.
- **Type annotation**: Explicitly specify the shape and data type of tensors in the function signature.
- **Direct call**: During testing, torch tensors and other types of variables can be passed directly without explicit conversion.
- **Dynamic shape support**: Works with `pypto.DYNAMIC` to support dimensions that vary at runtime.
- **Multiple run modes**: Supports both NPU and SIM (simulator) run modes.

## Prototype

```python
@pypto.frontend.jit(
    host_options=None,
    runtime_options=None,
    codegen_options=None,
    pass_options=None,
    verify_options=None,
    debug_options=None
)
def kernel_function(...):
    ...
```

## Parameters

| Parameter | Input/Output | Description |
|--------|----------|------|
| func | Input | Function decorated by frontend.jit, serving as the kernel entry, which describes the computation process and is used to build the computation graph. |
| host_options | Input | The type is `dict[str, any]`. Used to set host configuration items. For configuration item parameters, see [Parameters](./pypto-set_host_options.md). |
| runtime_options | Input | The type is `dict[str, any]`. Used to set runtime configuration items. For configuration item parameters, see [runtime_options parameters](#runtime_options_detail). |
| codegen_options | Input | The type is `dict[str, any]`. Used to set codegen configuration items. For configuration item parameters, see [Parameters](./pypto-set_codegen_options.md). |
| pass_options | Input | The type is `dict[str, any]`. Used to set pass configuration items. For configuration item parameters, see [Parameters](./pypto-set_pass_options.md). |
| verify_options | Input | The type is `dict[str, any]`. Used to set verify configuration items. For configuration item parameters, see [Parameters](./pypto-set_verify_options.md). |
| debug_options | Input | The type is `dict[str, any]`. Used to set debug configuration items. For configuration item parameters, see [Parameters](./pypto-set_debug_options.md). |

### runtime_options Parameters<a id="runtime_options_detail"></a>

| Parameter                         | Description                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| device_sched_mode               | Meaning: Sets the scheduling mode of the computation subgraph. <br> Description: 0: Default scheduling mode. Ready subgraphs are placed in a shared queue, and each scheduling thread preempts subgraphs for dispatch. Subgraph acquisition and dispatch follow first-in, first-out. <br> 1: L2cache affinity scheduling mode. The subgraph whose latest dependency becomes ready is dispatched first, achieving the effect of reusing the L2cache. <br> 2: Fair scheduling mode. When multiple threads on the AICPU schedule and manage multiple AICores, subgraph dispatch is controlled as much as possible to maintain fairness among multiple threads. This mode introduces additional scheduling management overhead. <br> 3: Enables both the L2cache affinity scheduling mode and the fair scheduling mode. <br> Type: int <br> Value Range: 0, 1, 2, or 3 <br> Default Value: 0 <br> Affected pass scope: N/A |
| stitch_function_max_num        | Meaning: Specifies the number of loop iterations built by a single stitch at runtime, thereby controlling the number of subgraphs scheduled and executed by the operator in a single dispatch. A larger value indicates higher parallelism. <br> Description: Intended to improve parallelism by increasing the number of loops built by stitch and reducing the number of interactions between stitch building and scheduling, without the user being aware of the memory usage limit. Use cases: single-operator development, where workspace has no explicit limit. When parallelism increases, the AICORE utilization is higher, but the runtime temporary memory usage also increases accordingly. When the configuration is too large, the stitch building overhead increases, which may cause scheduling to wait for stitch building, and the end-to-end performance may not necessarily be better. When no memory limit is set (`max_workspace_kb=0`), both parallelism and memory reservation are determined by this configuration. Once a valid `max_workspace_kb` is set, parallelism and memory are controlled by `max_workspace_kb`.<br> When parallelism already meets expectations and memory needs to be further reduced, use [pypto.experimental.set_runtime_options](./pypto-experimental-set_runtime_options.md). <br> Type: int <br> Value Range: 1 to 1024 <br> Default Value: 128 <br> Affected pass scope: N/A |
| max_workspace_kb               | Meaning: Sets the total upper limit for the workspace memory usage of the operator at runtime, in KB. <br> Description: Intended to compress the runtime workspace memory and parallelism according to the total upper limit when the operator execution has a workspace memory limit. `0` disables this option. When enabled, the value must be strictly greater than the minimum workspace required for the current operator to run; otherwise, a compilation error is reported. Use cases: whole-network deployment, insufficient NPU memory, or memory allocation failure. When this configuration takes effect, the temporary memory may be insufficient to support the target parallelism, the number of loops built by stitch decreases, and performance degrades. When this configuration is valid, memory reservation and parallelism are no longer calculated based on `stitch_function_max_num`. A configuration that is too large may increase memory usage or even cause OOM. When it is increased together with `device_sched_parallelism`, memory usually multiplies according to the parallelism.<br>To improve parallelism, use [pypto.experimental.set_runtime_options](./pypto-experimental-set_runtime_options.md). <br> Type: int <br> Value Range: 0 to 2147483647 <br> Default Value: 0 <br> Affected pass scope: N/A |
| run_mode                       | Meaning: Sets the execution device of the computation subgraph. <br> Description: <br> 0: Executes on the NPU. <br> 1: Executes on the simulator. <br> Type: int or `pypto.RunMode` enum <br> Value Range: 0 or 1 <br> Default Value: Determined by whether the CANN environment variable is set. If the environment variable is set, execution runs on the NPU; otherwise, execution runs on the simulator. <br> Affected pass scope: N/A |
| valid_shape_optimize            | Meaning: validshape compilation optimization option for dynamic shape scenarios. When this option is enabled, in the loop of a dynamic axis, the main block (where shape equals validshape) is compiled with static shape, and the tail block is compiled with dynamic shape. <br> Description: <br> 0: Default value, which disables the validshape compilation optimization option. All Loops are compiled with dynamic shape. <br> 1: Enables the validshape compilation optimization option. <br> Type: int <br> Value Range: 0 or 1 <br> Default Value: 0 <br> Affected pass scope: N/A |
| ready_on_host_tensors           | Meaning: Marks the list of input tensor names of the kernel entry function that are ready on the host side.<br> Description: If the computation logic of the operator has a value dependency on an input tensor (that is, the value of the tensor is obtained), and the device data of this tensor is already prepared on the host side, the CPU control flow can be dispatched in advance or executed on the host side to improve performance. This configuration item has two input forms: if the value of the value-dependent tensor cannot be obtained on the host side, you can configure the value-dependent operator names through ["tensor1", "tensor2", ...] to implement advance dispatch of the control flow; if the value of the value-dependent tensor can be obtained on the host side, you can pass the CPU tensor corresponding to the value-dependent tensor as an operator input, and configure the pairing relationship between the NPU tensor and the CPU tensor through [["tensor1_npu", "tensor1_cpu"], ["tensor2_npu", "tensor2_cpu"], ...], so that the framework performs control flow unrolling on the host side to improve performance.<br> Type: list of string or list of list of strings <br> Default Value: empty list <br> Affected pass scope: N/A |
| device_sched_parallelism        | Meaning: When pypto.loop in the operator has the parallel flag set (parallel=True), this configuration item specifies the parallelism of pypto.loop during scheduling execution. <br> Description: Before using this configuration item, ensure that there is no dependency between the iterations of the pypto.loop marked as parallel, satisfying the conditions for parallel scheduling. When the parallelism is greater than 1, multiple iteration tasks of this pypto.loop are scheduled and executed concurrently. Note that a larger parallelism value requires more workspace memory usage, which is usually proportional to the configured parallelism.<br> Type: int <br> Value Range: 1 to 8 <br> Default Value: 1 <br> Affected pass scope: N/A |
| launch_sched_aicpu_num        | Meaning: Specifies the number of Schedule AICPU threads to launch. <br> Description: When the specified number is greater than the maximum available AICPU count of the hardware or less than or equal to 0, the automatically calculated value is used. The maximum available AICPU count varies by model. For details, see [Constraints](#constraints).<br> Type: int <br> Value Range: 1 to 7 <br> Default Value: 7 <br> Affected pass scope: N/A |


## Return Value

Returns the decorated function, which can be called directly for execution.

## Constraints

1. Tensor parameters must be annotated with the `pypto.Tensor` type using type annotations.
2. Dynamic dimensions must be marked with `pypto.DYNAMIC` or `pypto.DYN` in the parameter annotations. If not marked, they are processed as static dimensions by default.
3. The tensor format is marked with **format**. **format** supports non-explicit marking (see `a` in Example 1), which defaults to `pypto.TileOpFormat.TILEOP_ND`.
   When **format** is explicitly marked, better performance can be achieved. The passed torch tensor must be consistent with the format declared by `pypto.Tensor` to obtain better performance.
4. Tensor parameters come first, followed by non-tensor parameters (such as `scalar` and `tiling`).
5. Non-tensor parameters support keyword arguments, positional arguments, and default values.
6. Maximum available AICPU count description:
   <!-- npu="950" id4 -->
   - For Ascend 950PR&950DT products, the maximum available AICPU count is 7 (the specific maximum count depends on the specific model).
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - For Atlas A3 products, the maximum available AICPU count is 5.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - For Atlas A2 products, the maximum available AICPU count is 5.
   <!-- end id6 -->

**Description of `pypto.Tensor[...]`**:

- In kernel functions, it is recommended to declare tensors using the `pypto.Tensor[[shape], dtype]` bracket syntax, which conforms to the Python type annotation specification.
- The legacy parenthesis syntax `pypto.Tensor([shape], dtype)` is also supported.
- Keyword arguments in the `key=value` form are not supported inside square brackets (due to Python syntax restrictions); they can only be passed by position or using a dictionary.
- `pypto.Tensor[]` (with empty arguments) is not supported.

## Examples

### Example 1: Basic Usage

```python
@pypto.frontend.jit
def add_kernel(
    a: pypto.Tensor([3], pypto.DT_FP32),
    b: pypto.Tensor([3], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_NZ),
    out: pypto.Tensor([3], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


# Directly pass a torch tensor for invocation.
x = torch.randn(3, dtype=torch.float32, device='npu:0')
y = torch.randn(3, dtype=torch.float32, device='npu:0')
result = add_kernel(x, y)
```

### Example 2: Specifying the Run Mode

```python
# NPU mode.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_npu(x: pypto.Tensor):
    ...

# Cost Model mode.
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x: pypto.Tensor):
    ...
```
