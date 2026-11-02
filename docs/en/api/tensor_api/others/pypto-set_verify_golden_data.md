# pypto.set\_verify\_golden\_data

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:52:51.848Z pushedAt=2026-10-28T11:57:30.182Z nl=2 -->

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

This API provides the following essential functions:

- Function 1: Sets the actual input and output lists used when the user executes an operator into the tool, so that the tool can use the same inputs for simulated computation during coarse-grained detection.
- Function 2: Sets the user's existing computation golden data into the tool, so that the tool can compare the simulated results with the golden outputs and thereby coarsely determine the correctness of the simulated results.

## Prototype

```python
set_verify_golden_data(in_out_tensors=None, goldens=None)
```

## Parameters

| Parameter | Input/Output | Description |
|-----------------|-----------|----------------------------------------------------------------------|
| in_out_tensors | Input | Meaning: Specifies the actual input and output lists used when the user executes the operator, and sets them into the detection tool in positional correspondence. <br> Description: This option does not need to be set in JIT compilation mode. <br> Type: List[Union(pypto.Tensor, torch.Tensor)] <br> Value Range: N/A <br> Default Value: N/A |
| goldens | Input | Meaning: Specifies the user's existing reference data (golden) output and sets it into the tool for comparison detection. <br> Description: This list has the same length and positional correspondence as the operator's input and output parameter lists. Its elements correspond to the operator's output parameters, and the positions of input parameters should be set to None. If a position is set to None, data comparison at that position is skipped. <br> Type: List[Union(pypto.Tensor, torch.Tensor)] <br> * The device attribute of torch.Tensor must be CPU; NPU is not supported. <br> Value Range: N/A <br> Default Value: N/A |

## Return Value

**void**: The **Set** method has no return value. The setting takes effect immediately upon success.

## Constraints

This function takes effect only after **pypto.set_verify_options(enable_pass_verify=True)** is set.

## Examples

```python
set_verify_golden_data(goldens=[None, None, golden_out0])
set_verify_golden_data([real_in0, real_in1, real_out0], [None, None, golden_out0])
```
