# pypto.Tensor.dim

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:03:18.108Z pushedAt=2026-10-28T11:57:30.272Z nl=2 -->

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

Obtains the dimension of a tensor.

## Prototype

```python
dim(self) -> int
```

## Parameters

None

## Return Value

Tensor dimension.

## Constraints

This is a read-only property.

## Examples

```python
t = pypto.tensor((2, 3, 4), pypto.DT_FP32)
out = t.dim
```

The results are as follows:

```python
Output data out: 3
```
