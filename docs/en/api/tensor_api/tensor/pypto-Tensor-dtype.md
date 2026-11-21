# pypto.Tensor.dtype

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

Obtains the data type of a tensor.

## Prototype

```python
dtype(self) -> DataType
```

## Parameters

None

## Return Value

Returns the data type of the tensor.

## Constraints

None

## Examples

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
out = t.dtype
```

Result example:

```python
Output data out: DataType.DT_FP32
```
