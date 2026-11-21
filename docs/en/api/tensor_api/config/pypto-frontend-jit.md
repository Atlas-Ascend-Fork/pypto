# pypto.frontend.jit

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Supported
<!-- end id8 -->
<!-- npu="9050" id35 -->
- Kirin 9050 processors: Supported
<!-- end id35 -->

## Description

`pypto.frontend.jit` is a core decorator in the frontend architecture. It is used to just-in-time (JIT) compile Python functions into efficient computational graphs and execute them on the NPU. The frontend does not support return values and only supports in-place modification. Torch tensors and variables of other types can be passed.

Main features:

- **In-place modification**: The kernel functions pass computation results by modifying output tensors in place, and return values are not supported.
- **Type annotation**: The shapes and data types of the tensors are specified in the function signature.
- **Direct calling**: During testing, torch tensors and variables of other types can be directly passed without explicit conversion.
- **Dynamic shape support**: It works with `pypto.DYNAMIC` to support runtime dimension changes.
- **Multiple running modes**: It supports two running modes: AI processor and SIM (simulator).

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

| Parameter| Input/Output| Description|
|--------|----------|------|
| func | Input| Function decorated by frontend.jit. It serves as the kernel entry and describes the computation process for constructing a computational graph.|
| host_options | Input| The type is `dict[str, any]`. It is used to set host configuration options. For details about the configuration options, see [Parameters](./pypto-set_host_options.md).|
| runtime_options | Input| The type is `dict[str, any]`. It is used to set runtime configuration options. For details about the configuration options, see [runtime_options Parameters](#runtime_options_detail).|
| codegen_options | Input| The type is `dict[str, any]`. It is used to set CodeGen configuration options. For details about the configuration options, see [Parameters](./pypto-set_codegen_options.md). |
| pass_options | Input| The type is `dict[str, any]`. It is used to set Pass configuration options. For details about the configuration options, see [Parameters](./pypto-set_pass_options.md). |
| verify_options | Input| The type is `dict[str, any]`. It is used to set Verify configuration options. For details about the configuration options, see [Parameters](./pypto-set_verify_options.md).|
| debug_options | Input| The type is `dict[str, any]`. It is used to set debug configuration options. For details about the configuration options, see [Parameters](./pypto-set_debug_options.md).|

### runtime_options Parameters <a id="runtime_options_detail"></a>

| Parameter                        | Description                                                        |
| ------------------------------ | ------------------------------------------------------------ |
| device_sched_mode               | Meaning: Sets the scheduling mode of the computational subgraph.<br> Description: 0: Default scheduling mode. Ready subgraphs are placed in a shared queue, and each scheduling thread preempts the subgraphs for dispatch. Subgraph acquisition and dispatch follow first-in, first-out.<br> 1: L2 cache affinity scheduling mode. The subgraph whose latest dependency becomes ready is dispatched first to reuse the L2 cache.<br> 2: Fair scheduling mode. When multiple threads on the AI CPU schedule and manage multiple AI Cores, the subgraphs are delivered to ensure fairness among the threads. This mode brings extra scheduling management overhead.<br> 3: Both the L2 cache affinity scheduling mode and fair scheduling mode are enabled.<br> Type: int<br> Value range: 0, 1, 2, or 3<br> Default value: 0<br> Affected Pass scope: N/A|
| stitch_function_max_num        | Meaning: Specifies the number of loop iterations built by Stitch at a time during operator execution. It controls the number of subgraphs scheduled and executed at a time. A larger value indicates higher parallelism.<br> Description: This configuration intends to increase the parallelism by increasing the number of loops built by Stitch and reducing the number of interactions between Stitch build and scheduling, without the user being aware of the memory usage limit. Scenario: single-operator development and lack of specific limit on the workspace. When parallelism increases, the AI Core usage is higher, but the temporary memory usage during operator execution increases accordingly. If the value is too large, the Stitch construction overhead increases, and the scheduling may wait for the Stitch construction. As a result, the end-to-end performance may not be improved. If the memory limit (`max_workspace_kb=0`) is not set, the parallelism and memory reservation are determined by this configuration. Once a valid `max_workspace_kb` is set, the parallelism and memory are controlled by `max_workspace_kb`.<br> If the parallelism meets the expectation and the memory needs to be reduced, use [pypto.experimental.set_runtime_options](./pypto-experimental-set_runtime_options.md).<br> Type: int<br> Value range: 1 to 1024<br> Default value: 128<br> Affected Pass scope: N/A|
| max_workspace_kb               | Meaning: Sets the total upper limit for the workspace memory usage of the operator at runtime, in KB.<br> Description: This configuration intends to reduce the workspace memory usage and parallelism during operator execution when the workspace memory is limited. `0`: This option is disabled. After this option is enabled, the value must be greater than the minimum Workspace required for running the current operator. Otherwise, a compilation error will be reported. Scenario: network-wide deployment, insufficient NPU memory, or memory allocation failure. After this configuration takes effect, the temporary memory may be insufficient to support the target parallelism. As a result, the number of loops built by Stitch decreases and the performance deteriorates. When this configuration is valid, the memory reservation and parallelism are no longer computed based on `stitch_function_max_num`. If this parameter is set to a large value, the memory usage may increase or even OOM may occur. If this parameter and `device_sched_parallelism` are set to large values, the memory usage increases exponentially with the parallelism.<br>To increase the parallelism, use [pypto.experimental.set_runtime_options](./pypto-experimental-set_runtime_options.md).<br> Type: int<br> Value range: 0 to 2147483647<br> Default value: 0<br> Affected Pass scope: N/A|
| run_mode                       | Meaning: Sets the device for executing a subgraph.<br> Description:<br> 0: The subgraph is executed on the NPU.<br> 1: The subgraph is executed on the simulator.<br> Type: int or `pypto.RunMode` enumeration<br> Value range: 0 or 1<br> Default value: Depending on whether the CANN environment variable is set. If the environment variable is set, the subgraph is executed on the NPU. Otherwise, the subgraph is executed on the simulator.<br> Affected Pass scope: N/A|
| valid_shape_optimize            | Meaning: Specifies the validshape compilation optimization option in dynamic shape scenarios. After this option is enabled, the main block (whose shape is the same as validshape) in the loop of a dynamic axis is compiled using the static shape, and the tail block is compiled using the dynamic shape.<br> Description:<br> 0: Default value, indicating that the validshape compilation optimization option is disabled. All loops are compiled using the dynamic shape.<br> 1: The validshape compilation optimization option is enabled.<br> Type: int<br> Value range: 0 or 1<br> Default value: 0<br> Affected Pass scope: N/A|
| ready_on_host_tensors           | Meaning: Marks the list of input tensor names of the kernel entry function that is ready on the host.<br> Description: If the computation logic of an operator has a value dependency on the value of an input tensor (which means the value of the tensor is obtained) and the device data of the tensor has been prepared on the host in advance, the CPU control flow can be transmitted in advance or executed on the host to improve performance. This configuration item can be set in two ways: If the value-dependent tensor cannot be obtained on the host, you can set this configuration item to ["tensor1", "tensor2", ...] to configure the names of the value-dependent operators to transmit the control flow in advance. If the value-dependent can be obtained on the host, you can use the CPU tensor corresponding to the value-dependent tensor as the operator input parameter, and set the mapping between the NPU tensor and the CPU tensor to [["tensor1_npu", "tensor1_cpu"], ["tensor2_npu", "tensor2_cpu"], ...], so that the framework can expand the control flow on the host to improve performance.<br> Type: list of strings or list of list of strings<br> Default value: empty list<br> Affected Pass scope: N/A|
| device_sched_parallelism        | Meaning: Specifies the parallelism for pypto.loop during scheduling and execution when parallel=True is set for pypto.loop in the operator.<br> Description: Before using this configuration item, ensure that there is no dependency between the iterations of pypto.loop marked as parallel, and the conditions for parallel scheduling are met. When the parallelism is greater than 1, multiple iterations of pypto.loop will be scheduled and executed concurrently. Note that a larger parallelism value requires more workspace memory, which is usually a multiple of the configured parallelism value.<br> Type: int<br> Value range: 1 to 8<br> Default value: 1<br> Affected Pass scope: N/A|
| launch_sched_aicpu_num        | Meaning: Specifies the number of Schedule AICPU threads to be started.<br> Description: If the specified number is greater than the maximum number of available AI CPUs or less than or equal to 0, the hardware automatically calculates the value. The maximum number of available AI CPUs varies depending on the model. For details, see [Constraints](#constraints).<br> Type: int<br> Value range: 1 to 7<br> Default value: 7<br> Affected Pass scope: N/A|


## Return Value

Returns the decorated function, which can be directly called and executed.

## Constraints

1. Tensor parameters must be annotated with the `pypto.Tensor` type using type annotations.
2. Dynamic dimensions must be marked with `pypto.DYNAMIC` or `pypto.DYN` in the parameter annotation. If not marked, they are processed as static dimensions by default.
3. The tensor format is marked with format. format supports non-explicit marking (see a in example 1) and defaults to pypto.TileOpFormat.TILEOP_ND.
   When format is explicitly marked, the performance is better. The passed torch tensor must be consistent with the format declared by pypto.Tensor for better performance.
4. Tensor parameters are placed before non-tensor parameters (such as `scalar` and `tiling`).
5. Non-tensor parameters support keyword parameters, positional parameters, and default values.
6. Restrictions on configuration items are as follows:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: A maximum of seven AI CPUs are available (the maximum number depends on the specific model). All runtime configuration items are supported. run_mode can be set to 0 (execution on the NPU) or 1 (execution on the simulator).
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: A maximum of five AI CPUs are available. All runtime configuration items are supported. run_mode can be set to 0 (execution on the NPU) or 1 (execution on the simulator).
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: A maximum of five AI CPUs are available. All runtime configuration items are supported. run_mode can be set to 0 (execution on the NPU) or 1 (execution on the simulator).
   <!-- end id6 -->
   <!-- npu="9030" id9 -->
   - Kirin 9030 processors: The maximum number of available AI CPUs is 0 (no AI CPU is available). The device_sched_mode, stitch_function_max_num, max_workspace_kb, ready_on_host_tensors, launch_sched_aicpu_num, and device_sched_parallelism parameters are not supported. run_mode can only be set to 1 (execution on the simulator). valid_shape_optimize is supported (in the Kirin scenario, dynamic shape is converted to static shape by default).
   <!-- end id9 -->
   <!-- npu="x90" id10 -->
   - Kirin X90 processors: The maximum number of available AI CPUs is 0 (no AI CPU is available). The device_sched_mode, stitch_function_max_num, max_workspace_kb, ready_on_host_tensors, launch_sched_aicpu_num, and device_sched_parallelism parameters are not supported. run_mode can only be set to 1 (execution on the simulator). valid_shape_optimize is supported (in the Kirin scenario, dynamic shape is converted to static shape by default).
   <!-- end id10 -->
   <!-- npu="9050" id11 -->
   - Kirin 9050 processors: The maximum number of available AI CPUs is 0 (no AI CPU is available). The device_sched_mode, stitch_function_max_num, max_workspace_kb, ready_on_host_tensors, launch_sched_aicpu_num, and device_sched_parallelism parameters are not supported. run_mode can only be set to 1 (execution on the simulator). valid_shape_optimize is supported (in the Kirin scenario, dynamic shape is converted to static shape by default).
   <!-- end id11 -->

**Description of pypto.Tensor[...]**:

- In kernel functions, the `pypto.Tensor[[shape], dtype]` bracket syntax is recommended, which conforms to the Python type annotation specifications.
- The legacy parenthesis syntax `pypto.Tensor([shape], dtype)` is also supported.
- The keyword arguments in the form of `key=value` are not supported inside square brackets (due to Python syntax restrictions). Only positional arguments or dictionaries can be used.
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


# Directly pass the torch tensor for calling.
x = torch.randn(3, dtype=torch.float32, device='npu:0')
y = torch.randn(3, dtype=torch.float32, device='npu:0')
result = add_kernel(x, y)
```

### Example 2: Specifying the Running Mode

```python
# NPU mode
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_npu(x: pypto.Tensor):
    ...

# Cost Model mode
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x: pypto.Tensor):
    ...
```
