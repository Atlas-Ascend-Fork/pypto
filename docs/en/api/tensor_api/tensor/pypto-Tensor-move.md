# pypto.Tensor.move

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:13:13.648Z pushedAt=2026-10-28T11:57:30.347Z nl=2 -->

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

Moves the data of one tensor to the current tensor.

## Prototype

```python
move(self, other: 'Tensor') -> None
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| other     | Input        | Source Tensor from which data is moved. |

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
