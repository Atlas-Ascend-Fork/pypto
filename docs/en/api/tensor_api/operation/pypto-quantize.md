# pypto.quantize

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

Converts high-precision floating-point data into a low-precision format. Currently, the following conversions are supported:

- Converts an input DT_FP32 tensor into a DT_INT8 tensor through symmetric quantization.
  $$
  \text{dst} = round(\text{input} * \text{scale})
  $$
- Converts an input DT_FP32 tensor into a DT_UINT8 tensor through asymmetric quantization.
  $$
  \text{dst} = round(\text{input} * \text{scale} + \text{zero\_points})
  $$

## Prototype

```python
quantize(input: Tensor, scale: Tensor, otype: DataType, axis: int, zero_points: Tensor) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).<br>The shape is denoted as [..., row, col].|
| scale  | Input     | Scaling factor.<br>Supported type: Tensor.<br>The tensor data type is the same as that of input, supporting DT_FP32.<br>Empty tensors are not supported.<br>The shape has one fewer dimension than the input, supporting only 1 to 3 dimensions.<br>The shape size must not exceed 2147483647 (INT32_MAX).<br>When axis = -1 or input.shape.size() -1, shape = [..., row].<br>When axis = -2 or input.shape.size() -2, shape = [..., col].|
| otype  | Input     | Numeric type of the returned value.<br>Currently, int8 and uint8 are supported, corresponding to symmetric quantization and asymmetric quantization, respectively.|
| axis  | Input     | Axis for quantization compression.<br>Currently, the last two axes are supported, that is, -1/-2 or input.shape.size() -1/input.shape.size()-2.<br>**When the input is 1D, only -1 is supported.**|
| zero_points  | Input     | Optional offset factor for asymmetric quantization.<br>Supported type: Tensor.<br>The tensor data type is the same as that of input, supporting DT_FP32.<br>Empty tensors are supported.<br>The shape has one fewer dimension than the input, supporting only 1 to 3 dimensions.<br>The shape size must not exceed 2147483647 (INT32_MAX).<br>When axis = -1 or input.shape.size() -1, shape = [..., row].<br>When axis = -2 or input.shape.size() -2, shape = [..., col].|

## Return Value

Returns the output tensor. Its data type is specified by otype, and its shape is the same as that of input.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([3, 4], pypto.DT_FP32)
scale = pypto.tensor([3], pypto.DT_FP32)
zero_points = pypto.tensor([3], pypto.DT_FP32)

# fp32 -> int8 symmetric quantization
y1 = pypto.quantize(x, scale, pypto.DT_INT8, -1, None)
# fp32 -> uint8 asymmetric quantization
y2 = pypto.quantize(x, scale, pypto.DT_UINT8, -1, zero_points)
```

Result example:

```python
Input  x:[[1.1, -2.2, 3.3, -4.4], [1.1, -2.2, 3.3, -4.4], [1.1, -2.2, 3.3, -4.4]]
Input  scale:[1.0, 1.0, 1.0]
Input zero_points:[-5.0, -5.0, -5.0]
Output y1:[[1, -2, 3, -4], [1, -2, 3, -4], [1, -2, 3, -4]]
Output y2:[[6, 3, 8, 1], [6, 3, 8, 1], [6, 3, 8, 1]]
```
