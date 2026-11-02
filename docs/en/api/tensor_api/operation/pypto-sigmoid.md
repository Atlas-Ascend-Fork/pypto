# pypto.sigmoid

<!-- md-trans-meta sourceCommit=74805bf4f8e98b142f26afa769f2f90aee900ccb translatedAt=2026-10-26T17:30:00.900Z pushedAt=2026-10-28T11:57:30.091Z nl=2 -->

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

Applies the sigmoid activation function to each element of the input tensor. The computation formula is as follows:t

$$
sigmoid(input) = \frac{1}{1 + e^{-input}}
$$

## Prototype

```python
sigmoid(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported data type: DT_FP32.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the results of mapping the input elements to the \(0, 1\) interval through the sigmoid function.

## Constraints

None

## Examples

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sigmoid(x)
```

The results are as follows:

```python
Input data x: [-3.0, 0.0, 2.0, 5.0]
Output data y: [0.0474, 0.5000, 0.8808, 0.9933]
```
