# pypto.Element.value

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

Obtains data.

## Prototype

```python
value(self) -> int | float
```

## Parameters

NA

## Return Value

Returns the data in the element.

## Constraints

Read-only attribute.

## Examples

```python
t = pypto.Element(pypto.DT_FP32, 3)
t.value
```
