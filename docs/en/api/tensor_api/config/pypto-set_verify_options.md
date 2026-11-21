# pypto.set\_verify\_options

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

Sets the self-check switch and function options for the precision debugging Verify feature.

## Prototype

```python
set_verify_options(*,
                   enable_pass_verify: Optional[bool] = None,
                   pass_verify_save_tensor: Optional[bool] = None,
                   pass_verify_save_tensor_dir: Optional[str] = None,
                   pass_verify_pass_filter: Optional[List[str]] = None,
                   pass_verify_error_tol: Optional[List[float]] = None,
                   ) -> None
```

## Parameters

| Parameter                         | Input/Output| Description                                                                |
|---------------------------------|-----------|----------------------------------------------------------------------|
| enable_pass_verify              | Input     | Meaning: Specifies whether all pass_verify_* options and APIs are valid. This is a general switch.<br> Description: True indicates that the function is enabled.<br> Type: bool<br> Value range: True or False<br> Default value: False|
| pass_verify_save_tensor         | Input     | Meaning: Specifies whether to save the operator-by-operator intermediate data during simulated computation to the disk.<br> Description: True indicates that the data is saved to the disk. The generation of the validation result directory and result file is not affected by this switch.<br> Type: bool<br> Value range: True or False<br> Default value: False|
| pass_verify_save_tensor_dir     | Input     | Meaning: Specifies the path for storing the validation result and data. The result file is generated in the verify subdirectory under the specified path.<br> Description: The value is a string that specifies an absolute path.<br> Type: str<br> Default value:<br> {RUNNING_DIR}/output/output_{TS} |
| pass_verify_pass_filter         | Input     | Meaning: Specifies the list of Pass names to be checked.<br> Description: The value is a valid Pass name.<br> If this parameter is not specified, Passes ["ExpandFunction", "ProcessAtomic", "L1CopyInReuseMerge", "InferDynShape", "PreGraphProcess", "InferParamIndex", "CodegenPreproc"] are checked by default. If this parameter is set to all, all Passes are checked. If this parameter is set to [], only tensor_graph is checked. If this parameter is set to an invalid name, the check is ignored.<br> Type: List[str]<br> Default value:<br> ["ExpandFunction", "ProcessAtomic", "L1CopyInReuseMerge", "InferDynShape", "PreGraphProcess", "InferParamIndex", "CodegenPreproc"] |
| pass_verify_error_tol           | Input     | Meaning: Specifies the rtol and atol required by the precision tool for precision comparison.<br> Description: The first value in the list is rtol, and the second value is atol. If the length of the list is not 2, the default values are used.<br> Type: List[float]<br> Default value: [1e-3, 1e-3]|

## Return Value

void: The Set method does not return any value. After the setting is successful, it takes effect during subsequent operator compilation.

## Constraints

- The setting must be performed before operator compilation and takes effect for the operators compiled subsequently.
- Once enabled, the validation is performed on the host during operator compilation, without changing the operator compilation and running results.
- When the switch is enabled (enable_pass_verify=True), the simulated computation library is automatically loaded. You do not need to manually load the library. In JIT mode, the framework loads the library during operator compilation. In non-JIT scenarios (for example, directly recording the pypto.function or IR compilation process), the library is loaded when this API is called. During the first loading, online compilation (depending on the Torch environment) is performed in the current process, which takes a long time.
- The validation result is not reported as an exception. That means the operator compilation is not interrupted regardless of whether the validation result is PASS or FAIL. You need to view the result files such as interpreter.log and verify_graph_result_brief.csv in the verify subdirectory of the pass_verify_save_tensor_dir directory ({RUNNING_DIR}/output/output_{TS} by default). ErrCode: FB4001 indicates that the verification results are inconsistent.

## Examples

```python
verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "/LARGE/DRIVE/DIR",
        }
pypto.set_verify_options(**verify_options)
```
