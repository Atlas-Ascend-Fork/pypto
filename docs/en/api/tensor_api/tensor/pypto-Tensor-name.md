# pypto.Tensor.name

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

Obtains or sets the name of a tensor.

## Prototype

```python
name(self) -> str
name(self, value: str) -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| value   | Input     | Returns the tensor name to be set.|

## Return Value

Returns the tensor name.

## Constraints

None

## Examples

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
n1 = t.name
t.name = "my_tensor"
n2 = t.name
```

Result example:

```python
Output n1: ""
Output n2: "my_tensor"
```
