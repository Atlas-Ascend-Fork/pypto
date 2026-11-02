# pypto.set\_verify\_options

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:37:52.106Z pushedAt=2026-10-28T11:57:29.620Z nl=2 -->

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

Sets the self-check switch of the precision debugging Verify feature and the functional options corresponding to the feature.

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

| Parameter | Input/Output | Description                                                                 |
|---------------------------------|-----------|----------------------------------------------------------------------|
| enable_pass_verify              | Input      | Meaning: Overall enable switch that determines whether all *pass_verify_* options and APIs take effect. <br> Description: True: enabled. <br> Type: bool <br> Value Range: True/False <br> Default Value: False |
| pass_verify_save_tensor         | Input      | Meaning: Whether to save the simulated computation data to disk. <br> Description: True: save to disk. <br> Type: bool <br> Value Range: True/False <br> Default Value: False |
| pass_verify_save_tensor_dir     | Input      | Meaning: Path for saving the check results and data. <br> Description: A string specifying an absolute path. <br> Type: str <br> Default Value: <br> {RUNNING_DIR}/output/output_{TS} |
| pass_verify_pass_filter         | Input      | Meaning: List of Pass names to be self-checked. <br> Description: Valid Pass names. If not specified, the following passes are checked by default: ["ExpandFunction", "ProcessAtomic", "L1CopyInReuseMerge", "InferDynShape", "PreGraphProcess", "InferParamIndex", "CodegenPreproc"]. If "all" is specified, all passes are checked. If [] is specified, no pass is checked and only tensor_graph is checked. If an invalid name is specified, it is ignored. <br> Type: List[str] <br> Default Value: empty |
| pass_verify_error_tol           | Input      | Meaning: rtol and atol used by the precision tool for precision comparison. <br> Description: The first value in the list is rtol and the second is atol. If the list length is not 2, the default value is used. <br> Type: List[float] <br> Default Value: [1e-3, 1e-3] |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

## Examples

```python
verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "/LARGE/DRIVE/DIR",
        }
pypto.set_verify_options(**verify_options)
```
