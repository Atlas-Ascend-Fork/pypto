# pypto.set\_verify\_golden\_data

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

This API provides the following essential functions:

- Function 1: Sets the actual input and output lists used by the user to execute the operator to the tool, so that the tool can use the same inputs for simulated computation during coarse-grained detection.
- Function 2: Sets the existing golden data of the user to the tool, so that the tool can compare the simulation results with the golden data after the computation is complete and coarsely determine the correctness of the simulation results.

## Prototype

```python
set_verify_golden_data(in_out_tensors=None, goldens=None)
```

## Parameters

| Parameter         | Input/Output| Description                                                                |
|-----------------|-----------|----------------------------------------------------------------------|
| in_out_tensors  | Input     | Definition: Sets the actual input and output lists used by the user to execute the operator to the detection tool in the same position.<br> Note: This option does not need to be set in JIT mode.<br> Type: List[Union(pypto.Tensor, torch.Tensor)]<br> Value range: N/A<br> Default value: N/A|
| goldens         | Input     | Definition: Sets the existing golden data output to the tool for comparison and detection.<br> Note: The length and position of this list are the same as those of the operator input and output parameter lists. The elements correspond to the output parameters of the operator. The input parameter position must be set to None. If the corresponding position is set to None, data comparison at this position is skipped.<br> Type: List[Union(pypto.Tensor, torch.Tensor)]<br> * The device attribute of torch.Tensor must be CPU. NPU is not supported.<br> Value range: N/A<br> Default value: N/A|

## Return Value

void: The Set method does not return any value. The setting takes effect immediately after the operation is successful.

## Constraints

This function only takes effect after pypto.set_verify_options(enable_pass_verify=True) is set.

## Examples

```python
set_verify_golden_data(goldens=[None, None, golden_out0])
set_verify_golden_data([real_in0, real_in1, real_out0], [None, None, golden_out0])
```
