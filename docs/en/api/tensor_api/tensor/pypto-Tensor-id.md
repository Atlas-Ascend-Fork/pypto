# pypto.Tensor.id

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:08:02.253Z pushedAt=2026-10-28T11:57:30.312Z nl=2 -->

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

Obtains the unique identifier of a tensor.

## Prototype

```python
id(self) -> int
```

## Parameters

None

## Return Value

Returns the unique identifier of a tensor.

## Constraints

This is a read-only property.

## Examples

```python
t = pypto.tensor((4, 4), pypto.DT_FP32)
print(t.id)  # Output the tensor ID.
```
