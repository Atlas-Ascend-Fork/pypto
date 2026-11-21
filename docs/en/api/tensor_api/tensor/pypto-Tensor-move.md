# pypto.Tensor.move

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

Moves the data of a tensor to the current tensor.

## Prototype

```python
move(self, other: 'Tensor') -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input     | Source tensor whose data is to be moved.|

## Return Value

None

## Constraints

None

## Examples

```python
t1 = pypto.tensor((2, 3), pypto.DT_FP32)
t2 = pypto.tensor((2, 3), pypto.DT_FP32)
# Move the data of t2 to t1.
t1.move(t2)
```
