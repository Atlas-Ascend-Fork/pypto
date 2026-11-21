# pypto.clone

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
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Copies the input source data and returns the copy.

## Prototype

```python
clone(input: Tensor) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>The supported data types are those supported by PyPTO.<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor with the same shape and data type as the input.

## Constraints

1. The input and output tensors must have the same type.

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.clone(x)
```

Result example:

```python
input x: [[1, 2],
          [3, 4]]
output y: [[1, 2],
           [3, 4]]
```
