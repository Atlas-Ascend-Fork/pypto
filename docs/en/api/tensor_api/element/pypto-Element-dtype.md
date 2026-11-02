# pypto.Element.dtype

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:52:41.823Z pushedAt=2026-10-28T11:57:29.715Z nl=2 -->

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

Obtains the data type.

## Prototype

```python
dtype(self) -> pypto.DataType
```

## Parameters

NA

## Return Value

Returns the data type.

## Constraints

Read-only data.

## Examples

```python
t = pypto.Element(pypto.DT_FP32, 3)
t.dtype
```
