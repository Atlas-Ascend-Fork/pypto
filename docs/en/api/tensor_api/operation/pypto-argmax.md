# pypto.argmax

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

Computes the index of the maximum value of a multi-dimensional vector along a specified dimension.

Defines the specified computation dimension as the R-axis (Reduce axis) and the non-specified dimension as the A-axis (Normal axis). The operation is performed on a two-dimensional matrix with shape \(2, 3\). If the maximum value index is computed along the first dimension, the output result is \[1, 1, 1\]. If the maximum value index is computed along the second dimension, the output result is \[2, 2\].

## Prototype

```python
argmax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Dimension for reduction.<br>Any single axis is supported.                                      |
| keepdim | Input     | Whether to retain the reduced dimension after reduction.<br>The default value is False.|

## Return Value

Returns the output tensor. Its data type is DT_INT32, and its shape depends on the keepdim parameter.

If keepdim is set to True, the reduced dimension is retained after the reduction operation. The shape of the output tensor is the same as that of the input tensor in all dimensions except the dimension specified by dim, where the size is 1.

If keepdim is set to False (default), the reduced dimension is removed from the output tensor, but the corresponding dimension in TileShape remains unchanged. Therefore, reset TileShape before calling other operations.

## Constraints

1. The size of TileShape must not exceed 64 KB.

2. The last axis must be 32-byte aligned.

3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the input.

For example, if the input shape is [m, n], the output is [m, 1], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Note: If keepdim is set to false, the reduced dimension is removed from the output tensor, but the corresponding dimension in TileShape remains unchanged. Therefore, reset TileShape before calling other operations.

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argmax(x, -1, True)
```

Result example:

```python
Input data x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Output data y: [[2],
             [2]]
```
