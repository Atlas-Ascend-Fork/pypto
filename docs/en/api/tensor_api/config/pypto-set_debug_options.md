# pypto.set\_debug\_options

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
<!-- npu="9050" id9 -->
- Kirin 9050 processors: Supported
<!-- end id9 -->

## Description

Sets the debug options.

## Prototype

```python
set_debug_options(*,
                  compile_debug_mode: Optional[int] = None,
                  runtime_debug_mode: Optional[int] = None,
                  dump_pass_graph: Optional[List[str]] = None,
                  ) -> None
```

## Parameters

| Parameter              | Input/Output| Description                                                                |
|----------------------|-----------|----------------------------------------------------------------------|
| compile_debug_mode   | Input     | Meaning: Sets the debug mode in the compilation phase.<br> Description: 0: The debug mode in the compilation phase is disabled by default.<br> 1: The debug mode in the graph compilation phase is enabled. This mode enables the configuration of graph compilation in one click. Currently, only computational graphs are supported.<br> 2: The fixed CCE function is enabled. The code output path on the device side is fixed in one-click mode, and the existing CCE file is not overwritten. The number of concurrent compilation threads is still controlled by `codegen.parallel_compile`. The kernel name is numbered based on the task identity. The multi-thread compilation product can be reproduced.<br> Type: int<br> Value range: 0, 1, or 2<br> Default value: 0.<br> Affected Pass scope: N/A|
| runtime_debug_mode   | Input     | Meaning: Sets the debug mode in the execution phase.<br> Description: 0: The debug mode in the execution phase is disabled by default.<br> 1: The debug mode in the execution phase is enabled. This mode enables the configuration of graph execution in one click. Currently, only swimlane diagrams are supported.<br> 2: The AI Core model (AICORE_MODEL) simulation mode is enabled.<br> 3: The data dump for runtime dependency correctness validation is enabled.<br> 4: The GM memory out-of-bounds check at runtime is enabled. In addition, at the entry of the control flow, a runtime integer division assertion is generated for each `assume_divisible` hypothesis expression that you have explicitly registered and normalized. The generated code uses the actual runtime value to compute the expression. The compiler inference expression that is not explicitly registered is not checked.<br> Type: int<br> Value range: 0, 1, 2, 3, or 4<br> Default value: 0<br> Affected Pass scope: N/A|
| dump_pass_graph      | Input     | Meaning: Enables computational graph dump and IR printing for one or more Passes.<br> Description: An empty list indicates that the function is disabled. If this parameter is set to a Pass identifier list, computational graph dump and IR printing are enabled only for the matched Passes. For other Passes, no computational graph or IR is output. If a Pass identifier that does not exist is configured, the identifier does not match any Pass, and an alarm is generated.<br> Type: List[str]<br> Default value: []<br> Affected Pass scope: specified Pass|

## Return Value

void: The Set method does not return any value. The setting takes effect immediately after the operation is successful.

## Constraints

- When compile_debug_mode is set to 2 (fixed CCE), the code output path on the device is determined by the environment variable ASCEND_WORK_PATH (`$ASCEND_WORK_PATH/pypto/<name>`). If ASCEND_WORK_PATH is not set, the code is output to the current working directory. In this mode, TILE_FWK_OUTPUT_DIR is not involved in the computation of the device code path.
- The setting of compile_debug_mode=2 is equivalent to enabling `codegen.fixed_output_path` and disabling `codegen.force_overwrite` at the same time. The existing CCE file will not be overwritten, which facilitates repeated modification of CCE debugging in a fixed path.
- The dump_pass_graph configuration item is set to the Pass identifier in the current Pass strategy, for example, "RemoveRedundantReshape", instead of the Pass name enumeration value in C++.
- dump_pass_graph supports the configuration of multiple Pass identifiers. The identifiers that do not match the current Pass strategy in the list will be ignored, and an alarm will be generated. If none of the identifiers in the list is matched, the computational graph or IR of any Pass will not be output.
- If both compile_debug_mode=1 and dump_pass_graph are set, compile_debug_mode=1 takes precedence. In this case, the computational graph dump and IR printing of all Passes are enabled.
<!-- npu="x90,9030,9050" id6 -->
- runtime_debug_mode usage:
    <!-- npu="9030" id7 -->
    - Kirin 9030 processors: Setting runtime_debug_mode to 1, 3, or 4 is not supported.
    <!-- end id7 -->
    <!-- npu="x90" id8 -->
    - Kirin X90 processors: Setting runtime_debug_mode to 1, 3, or 4 is not supported.
    <!-- end id8 -->
    <!-- npu="9050" id10 -->
    - Kirin 9050 processors: Setting runtime_debug_mode to 1, 3, or 4 is not supported.
    <!-- end id10 -->
<!-- end id6 -->

## Examples

```python
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)
pypto.set_debug_options(compile_debug_mode=2)
pypto.set_debug_options(dump_pass_graph=["RemoveRedundantReshape", "InferMemoryConflict"])
```
