# pypto.unsqueeze

<!-- md-trans-meta sourceCommit=d273c675dccb303668218cb882e7548a01692832 translatedAt=2026-10-26T17:46:30.767Z pushedAt=2026-10-28T11:57:30.151Z nl=2 -->

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

Adds a dimension to the input tensor.

## Prototype

```python
unsqueeze(input: Tensor, dim: int) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| input | Input | Source operand.<br>Supported data types: data types supported by PyPTO.<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| dim | Input | Position (index) at which the new dimension is inserted.<br>Negative indexes are supported.<br>Must be within the range [-input.dim - 1, input.dim]. |

## Constraints

The input tensor does not support dynamic axes. That is, no axis in the shape of input can be marked as `pypto.DYNAMIC`.

## Return Value

Returns the output tensor with a new dimension of size 1 added at the specified dimension **dim**. The output tensor shares data with the input tensor and has identical attributes.

## Examples

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.unsqueeze(x, 0)
```

The results are as follows:

```python
Output data x: [[1, 2, 3],
            [4, 5, 6]]
Output data y: [[[1, 2, 3],
             [4, 5, 6]]]
```
