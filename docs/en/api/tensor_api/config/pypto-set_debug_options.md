# pypto.set\_debug\_options

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:31:41.985Z pushedAt=2026-10-28T11:57:29.598Z nl=2 -->

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

| Parameter       | Input/Output | Description                                                          |
|----------------------|--------------|----------------------------------------------------------------------|
| compile_debug_mode   | Input        | Meaning: Sets the compilation-phase debug mode. <br> Description: 0: indicates that the compilation-phase debug mode is disabled by default. <br> 1: indicates that the graph compilation-phase debug mode is enabled, which enables graph compilation-related configurations with one click. Currently, this includes only the compute graph. <br> 2: indicates that the fixed CCE function is enabled, which enables the fixed device-side code output path with one click and does not overwrite existing CCE files. The number of parallel compilation threads is still controlled by `codegen.parallel_compile`, kernel names are numbered deterministically by task identity, and multi-threaded compilation artifacts are reproducible. <br> Type: int <br> Value range: 0, 1, or 2 <br> Default value: 0 <br> Impacted Pass scope: NA |
| runtime_debug_mode   | Input        | Meaning: Sets the execution-phase debug mode. <br> Description: 0: indicates that the execution-phase debug mode is disabled by default. <br> 1: indicates that the execution-phase debug mode is enabled, which enables graph execution-related configurations with one click. Currently, this includes only the swimlane graph. <br> 2: indicates that the AICORE_MODEL simulation mode is enabled. <br> 3: indicates that runtime dependency correctness verification data dump is enabled. <br> 4: indicates that runtime GM memory out-of-bounds checking is enabled, and at control flow entry points, a runtime divisibility assertion is generated for each `assume_divisible` assumption expression that is explicitly registered and normalized by the user. The generated code computes the expression using actual runtime values, and compiler-derived expressions that are not explicitly registered are not checked. <br> Type: int <br> Value range: 0, 1, 2, 3, or 4 <br> Default value: 0 <br> Impacted Pass scope: NA |
| dump_pass_graph      | Input        | Meaning: Specifies one or more Passes for which compute graph dump and IR printing are enabled. <br> Description: An empty list indicates disabled. When configured as a list of Pass identifiers, compute graph dump and IR printing are enabled only for matching Passes, and other Passes do not output compute graphs or IR. If a Pass identifier that does not exist is configured, that identifier does not match any Pass and a warning is output. <br> Type: List[str] <br> Default value: [] <br> Impacted Pass scope: Specified Passes |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

- When **compile_debug_mode** is set to **2** (fixed CCE), the device-side code output path is determined by the environment variable **ASCEND_WORK_PATH** (`$ASCEND_WORK_PATH/pypto/<name>`). If **ASCEND_WORK_PATH** is not set, the code is output to the current working directory. In this mode, **TILE_FWK_OUTPUT_DIR** does not participate in the device-side code path computation.
- compile_debug_mode=2 is equivalent to enabling `codegen.fixed_output_path` and disabling `codegen.force_overwrite` at the same time. Existing CCE files are not overwritten, which facilitates repeatedly modifying CCE for debugging under a fixed path.
- The dump_pass_graph configuration item takes the Pass identifiers in the current Pass policy, for example "RemoveRedundantReshape", not the PassName enumeration values in C++.
- dump_pass_graph supports configuring multiple Pass identifiers. Identifiers in the list that do not match the current Pass policy are ignored and a warning is output. If none of the identifiers in the list match, no compute graph or IR of any Pass is output.
- When compile_debug_mode=1 and dump_pass_graph are set at the same time, compile_debug_mode=1 takes precedence, and compute graph dump and IR printing are ultimately enabled for all Passes.

## Examples

```python
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)
pypto.set_debug_options(compile_debug_mode=2)
pypto.set_debug_options(dump_pass_graph=["RemoveRedundantReshape", "InferMemoryConflict"])
```
