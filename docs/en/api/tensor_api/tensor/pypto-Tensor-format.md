# pypto.Tensor.format

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:05:44.265Z pushedAt=2026-10-28T11:57:30.302Z nl=2 -->

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

Obtains the format of a tensor.

## Prototype

```python
format(self) -> TileOpFormat
```

## Parameters

None

## Return Value

**TileOpFormat**: Returns the format of a tensor.

## Constraints

This is a read-only property.

## Examples

```python
t = pypto.tensor((4, 4), pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND)
print(t.format)
```

The results are as follows:

```text
Output: TileOpFormat.TILEOP_ND
```
