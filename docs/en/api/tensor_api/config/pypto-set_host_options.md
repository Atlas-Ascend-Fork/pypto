# pypto.set\_host\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:34:42.723Z pushedAt=2026-10-28T11:57:29.604Z nl=2 -->

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

This API is the core part of the runtime dynamic configuration management feature provided by the compilation framework. It converts parameters originally statically configured in **tile_fwk_config.json** into dynamic, programmable instructions, primarily to control the execution of the on-board process.

## Prototype

```python
set_host_options(*, compile_stage: Optional[CompStage] = None,
                    compile_monitor_enable: Optional[int] = None,
                    compile_timeout: Optional[int] = None,
                    compile_timeout_stage: Optional[int] = None,
                    compile_monitor_print_interval: Optional[int] = None) -> None
```

## Parameters

| Parameter          | Input/Output | Description                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| compile_stage    | Input      | Meaning: Controls the compilation phase to execute. <br> Description: <br> ALL_COMPLETE: No effect; compilation and execution proceed normally. <br> TENSOR_GRAPH: Compilation phase; stops after generating the final tensor graph. <br> TILE_GRAPH: Compilation phase; stops after generating the final tile graph. <br> EXECUTE_GRAPH: Compilation phase; stops after generating the final execution graph. <br> CODEGEN_INSTRUCTION: Compilation phase; stops after generating instruction code. <br> CODEGEN_BINARY: Stops after generating the code binary, ending the compilation phase. <br> Value range: CompStage (ALL_COMPLETE/TENSOR_GRAPH/TILE_GRAPH/EXECUTE_GRAPH/CODEGEN_INSTRUCTION/CODEGEN_BINARY) <br> Default value: ALL_COMPLETE |
| compile_monitor_enable    | Input      | Meaning: Controls the compilation phase monitoring mode (Watchdog mechanism). <br> Description: <br> 0: Disables Compiler Monitor. <br> 1: Enables Compiler Monitor and disables Pass details. At compilation start, the timeout threshold is recorded in the INFO log; when the total compilation duration exceeds the threshold, a timeout summary is recorded in the WARNING log. <br> 2: Enables Compiler Monitor and enables Pass details. In addition to the behavior of mode 1, Pass progress is printed on the screen during compilation, and a WARNING is printed when a phase times out. <br> Value range: int [0, 2] <br> Default value: 1 |
| compile_timeout    | Input      | Meaning: Sets the total compilation duration timeout threshold. When the total duration of the current compilation exceeds this value, a timeout warning message is printed. <br> Description: Takes effect only when compile_monitor_enable is 1 or 2. Unit: seconds. A value of 0 disables warning message printing. <br> Numeric type: int. <br> Value range: int [0, 2147483647] <br> Default value: 600 |
| compile_timeout_stage    | Input      | Meaning: Sets the single-phase compilation duration timeout threshold. When the duration of a single phase in the compilation process exceeds this value, a timeout warning message is printed. <br> Description: Takes effect only when compile_monitor_enable is 1 or 2. Unit: seconds. A value of 0 disables warning message printing. <br> Numeric type: int. <br> Value range: int [0, 2147483647] <br> Default value: 0 (disabled) |
| compile_monitor_print_interval    | Input      | Meaning: Sets the compilation progress printing interval. When the duration of a single phase in the compilation process exceeds 60s, progress printing starts at this interval. <br> Description: Takes effect only when compile_monitor_enable is 1 or 2. Unit: seconds. <br> Numeric type: int. <br> Value range: int [0, 2147483647] <br> Default value: 60 |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

- Type safety: Ensure that the type of the passed parameter value exactly matches the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
- Scope: Parameter settings are global and affect all subsequent compilation processes.

## Examples

```python
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH,
                       compile_monitor_enable=1,
                       compile_timeout=120,
                       compile_timeout_stage=30,
                       compile_monitor_print_interval=20
                       )
```
