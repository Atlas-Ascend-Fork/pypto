# pypto.one\_hot

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
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Converts an integer tensor into the corresponding one-hot encoding, where each integer is converted into a vector with only the corresponding position set to 1 and all other positions set to 0.

## Prototype

```python
one_hot(input: Tensor, num_classes: int) -> Tensor
```

## Parameters

| Parameter     | Input/Output| Description                                                                |
|-------------|-----------|----------------------------------------------------------------------|
| input       | Input     | Source operand.<br>Supported type: Tensor.<br>The tensor supports the following data types: DT_INT8, DT_INT16, DT_INT32, and DT_INT64.<br>1 to 3 dimensions are supported.<br>Internal elements must be non-negative.<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| num_classes | Input     | Length of the one-hot encoding.<br>It must be greater than the maximum element in the input.|

## Return Value

Returns a tensor of shape (input, num\_classes) with data type DT\_INT64.

## Constraints

TileShape is used to tile the output. Its dimensions must be the same as those of the output, and its last axis must equal num\_classes.

Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n, t], where t = num_classes, and TileShape is set to [m1, n1, t1], then m1 and n1 are used to tile the m and n axes, respectively. t1 must equal num_classes. The t axis cannot be tiled and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_INT32)
y = pypto.one_hot(x, 5)
```

Result example:

```python
Input data x: [0, 2, 4]
Output data y: [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]
```
