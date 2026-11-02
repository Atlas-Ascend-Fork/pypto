# pypto.bytes\_of

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:48:12.448Z pushedAt=2026-10-28T11:57:29.673Z nl=2 -->

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

Returns the size in bytes occupied by a data type.

## Prototype

```python
bytes_of(dtype: pypto.DataType) -> int
```

## Parameters

| Parameter | Input/Output | Description                              |
|-----------|--------------|------------------------------------------|
| dtype     | Input        | Data type whose size in bytes is to be queried. |

## Return Value

Returns the size in bytes occupied by the data type.

## Constraints

None

## Examples

```python
pypto.bytes_of(pypto.DT_FP32)
```

The results are as follows:

```python
Output: 4
```
