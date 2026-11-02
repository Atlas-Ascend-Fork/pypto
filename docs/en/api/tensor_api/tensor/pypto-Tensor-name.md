# pypto.Tensor.name

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:14:32.280Z pushedAt=2026-10-28T11:57:30.353Z nl=2 -->

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

Obtains or sets the name of a tensor.

## Prototype

```python
name(self) -> str
name(self, value: str) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| value | Input | Name of the Tensor to set. |

## Return Value

Name of the tensor.

## Constraints

None

## Examples

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
n1 = t.name
t.name = "my_tensor"
n2 = t.name
```

The results are as follows:

```python
Output n1: ""
Output n2: "my_tensor"
```
