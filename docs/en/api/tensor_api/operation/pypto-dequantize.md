# pypto.dequantize

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:23:26.390Z pushedAt=2026-10-28T11:57:29.846Z nl=2 -->

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

Converts quantized low-precision data to a high-precision format and applies the **scale** and **zero_points** parameters. Currently supports:

- Dequantizes an input tensor of DT_INT8/DT_INT16 into a tensor of DT_FP32.
  $$
  \text{dst} = ([float]\text{input} + \text{zero\_points}) * \text{scale}
  $$

## Prototype

```python
dequantize(input: Tensor, scale: Tensor, otype: DataType, axis: int, zero_points: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT8/DT_INT16.<br>Empty tensors are not supported.<br>The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**).<br>The shape is denoted as [..., row, col]. |
| scale  | Input      | Scaling factor.<br>Supported type: Tensor.<br>The tensor data type must match `otype` and supports DT_FP32.<br>Empty tensors are not supported.<br>The shape has one fewer dimension than `input`, supporting only 1 to 3 dimensions.<br>The shape size must not exceed **2147483647** (**INT32_MAX**).<br>When axis = -1 or input.shape.size() -1, shape = [..., row].<br>When axis = -2 or input.shape.size() -2, shape = [..., col].|
| otype  | Input      | Numeric type of the return value.<br>Currently, DT_FP32 is supported.|
| axis  | Input      | Axis for dequantization compression.<br>Currently, the last two axes are supported, that is, -1/-2 or input.shape.size() -1/input.shape.size()-2.<br>**When input is 1D, only -1 is supported.** |
| zero_points  | Input      | Optional offset factor for asymmetric quantization.<br>Supported type: Tensor.<br>The tensor data type must match `otype` and supports DT_FP32.<br>Empty tensors are supported.<br>The shape has one fewer dimension than `input`, supporting only 1 to 3 dimensions.<br>The shape size must not exceed **2147483647** (**INT32_MAX**).<br>When axis = -1 or input.shape.size() -1, shape = [..., row].<br>When axis = -2 or input.shape.size() -2, shape = [..., col].|

## Return Value

Returns the output tensor, whose data type is specified by **otype** and whose shape is the same as **input**.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n] and the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([3, 4], pypto.DT_INT8)
scale = pypto.tensor([3, 1], pypto.DT_FP32)
zero_points = pypto.tensor([3, 1], pypto.DT_FP32)

# fp32 -> int8 symmetric dequantization
y1 = pypto.dequantize(x, scale, pypto.DT_FP32, -1, None)
# fp32 -> uint8 asymmetric dequantization
y2 = pypto.dequantize(x, scale, pypto.DT_FP32, -1, zero_points)
```

The results are as follows:

```python
Input  x:[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
Input  scale:[1.0, 1.0, 1.0]
Input  zero_points:[-2.0, -2.0, -2.0]
Output y1:[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
Output y2:[[-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0]]
```
