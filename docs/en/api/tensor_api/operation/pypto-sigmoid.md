# pypto.sigmoid

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

Applies the sigmoid activation function to each element of the input tensor. The computation formula is as follows:

$$
sigmoid(input) = \frac{1}{1 + e^{-input}}
$$

## Prototype

```python
sigmoid(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported data type: DT_FP32. <!-- npu="x90,9030,9050" id7 -->Kirin processors: DT_FP16 and DT_FP32.<!-- end id7 --><br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the results of mapping the input elements to the (0, 1) interval using the sigmoid function.

## Constraints

None

## Examples

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sigmoid(x)
```

Result example:

```python
Input data x: [-3.0, 0.0, 2.0, 5.0]
Output data y: [0.0474, 0.5000, 0.8808, 0.9933]
```
