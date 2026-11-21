# pypto.unsqueeze

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
<!-- npu="9050" id10 -->
- Kirin 9050 processors: Supported
<!-- end id10 -->

## Description

Adds a dimension to the input tensor.

## Prototype

```python
unsqueeze(input: Tensor, dim: int) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>The supported data types are those supported by PyPTO. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id9 --><br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Position (index) where the new dimension is inserted.<br>Negative indexes are supported.<br>The value must be within the range of [-input.dim - 1, input.dim].|

## Constraints

The input tensor does not support dynamic axes. That is, no axis in the shape of input can be marked as `pypto.DYNAMIC`.

<!-- npu="x90,9030,9050" id6 -->
1. A tensor supports the following data types:
   <!-- npu="9030" id7 -->
   - Kirin 9030 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id7 -->
   <!-- npu="x90" id8 -->
   - Kirin X90 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id8 -->
   <!-- npu="9050" id11 -->
   - Kirin 9050 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id11 -->
<!-- end id6 -->

## Return Value

Returns the output tensor with a new dimension of size 1 added at the dimension specified by dim. The output tensor shares data, and has identical attributes, with the input tensor.

## Examples

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.unsqueeze(x, 0)
```

Result example:

```python
Output data x: [[1, 2, 3],
            [4, 5, 6]]
Output data y: [[[1, 2, 3],
             [4, 5, 6]]]
```
