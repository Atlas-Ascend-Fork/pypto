# pypto.clone

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:15:52.172Z pushedAt=2026-10-28T11:57:29.814Z nl=2 -->

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

Copies the input source data and returns the copy.

## Prototype

```python
clone(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported data types: data types supported by PyPTO.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor with the same shape and data type as the input.

## Constraints

1. The input tensor and the output tensor must have the same type.

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.clone(x)
```

The results are as follows:

```python
input x: [[1, 2],
          [3, 4]]
output y: [[1, 2],
           [3, 4]]
```
