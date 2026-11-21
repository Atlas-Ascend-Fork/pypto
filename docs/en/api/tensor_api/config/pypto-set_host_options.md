# pypto.set\_host\_options

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
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

This API is the core part of the runtime dynamic configuration management feature provided by the compilation framework. It converts the parameters that were originally statically configured in the tile\_fwk\_config.json file into dynamic, programmable instructions. The main function is to control the on-board execution flow.

## Prototype

```python
set_host_options(*, compile_stage: Optional[CompStage] = None,
                    compile_monitor_enable: Optional[int] = None,
                    compile_timeout: Optional[int] = None,
                    compile_timeout_stage: Optional[int] = None,
                    compile_monitor_print_interval: Optional[int] = None) -> None
```

## Parameters

| Parameter         | Input/Output| Description                                                                |
|-----------------|-----------|----------------------------------------------------------------------|
| compile_stage    | Input     | Meaning: Controls the compilation execution phase.<br> Note:<br> ALL_COMPLETE: The compilation and running are normal.<br> TENSOR_GRAPH: The compilation ends after the final tensor graph is generated.<br> TILE_GRAPH: The compilation ends after the final tile graph is generated.<br> EXECUTE_GRAPH: The compilation stops after the final execution graph is generated.<br> CODEGEN_INSTRUCTION: The compilation ends after the instruction code is generated.<br> CODEGEN_BINARY: The compilation ends after the binary code is generated.<br> Value range: CompStage (ALL_COMPLETE/TENSOR_GRAPH/TILE_GRAPH/EXECUTE_GRAPH/CODEGEN_INSTRUCTION/CODEGEN_BINARY)<br> Default value: ALL_COMPLETE|
| compile_monitor_enable    | Input     | Meaning: Controls the monitoring mode (watchdog mechanism) in the compilation phase.<br> Note:<br> 0: The compiler monitor is disabled.<br> 1: The compiler monitor is enabled, and the Pass details are disabled. When the compilation starts, the timeout threshold is recorded in the INFO log. When the total compilation time exceeds the threshold, the timeout summary is recorded in the WARNING log.<br> 2: The compiler monitor is enabled, and the Pass details are enabled. In addition to mode 1, the Pass progress is printed on the screen during the compilation, and a WARNING message is printed when the compilation times out.<br> Value range: int [0, 2]<br> Default value: 1|
| compile_timeout    | Input     | Meaning: Specifies the timeout threshold for the total compilation time. If the total compilation time exceeds the value of this parameter, a timeout alarm is printed.<br> Description: This parameter is valid only when compile_monitor_enable is set to 1 or 2. The unit is second. If the value is 0, alarm printing is disabled.<br> Value type: int<br> Value range: int [0, 2147483647]<br> Default value: 600|
| compile_timeout_stage    | Input     | Meaning: Specifies the timeout threshold for the time consumed by a single compilation phase. If the time consumed by a single compilation phase exceeds the value of this parameter, a timeout alarm is printed.<br> Description: This parameter is valid only when compile_monitor_enable is set to 1 or 2. The unit is second. If the value is 0, alarm printing is disabled.<br> Value type: int<br> Value range: int [0, 2147483647]<br> Default value: 0 (disabled)|
| compile_monitor_print_interval    | Input     | Meaning: Specifies the interval for printing the compilation progress. If the time consumed by a single compilation phase exceeds 60s, the progress is printed at this interval.<br> Description: This parameter is valid only when compile_monitor_enable is set to 1 or 2. The unit is second.<br> Value type: int<br> Value range: int [0, 2147483647]<br> Default value: 60|

## Return Value

void: The Set method does not return any value. The setting takes effect immediately after the operation is successful.

## Constraints

- Type safety: Ensure that the types of the passed parameter values are the same as those defined for the parameters. Otherwise, undefined behavior or runtime errors may occur.
- Scope: The parameter settings are global and affect all subsequent compilation processes.

## Examples

```python
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH,
                       compile_monitor_enable=1,
                       compile_timeout=120,
                       compile_timeout_stage=30,
                       compile_monitor_print_interval=20
                       )
```
