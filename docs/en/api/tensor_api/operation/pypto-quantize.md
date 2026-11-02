# pypto.quantize

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:20:14.721Z pushedAt=2026-10-28T11:57:30.050Z nl=2 -->

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

Converts high-precision floating-point data into a low-precision format. Currently, the following are supported:

- Converts an input **DT_FP32** tensor into a **DT_INT8** tensor through symmetric quantization.
  $$
  \text{dst} = round(\text{input} * \text{scale})
  $$
- Converts an input **DT_FP32** tensor into a **DT_UINT8** tensor through asymmetric quantization.
  $$
  \text{dst} = round(\text{input} * \text{scale} + \text{zero\_points})
  $$

## Prototype

```python
quantize(input: Tensor, scale: Tensor, otype: DataType, axis: int, zero_points: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_FP32.<br>Empty tensors are not supported. The shape only supports 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).<br>The shape is denoted as [..., row, col]. |
| scale  | Input      | Scaling factor.<br>Supported type: Tensor.<br>The tensor data type is the same as that of **input**, supporting **DT_FP32**.<br>Empty tensors are not supported.<br>The shape has one fewer dimension than **input**, only supporting 1 to 3 dimensions.<br>The shape size must not exceed **2147483647** (that is, **INT32_MAX**).<br>When **axis** = -1 or input.shape.size() - 1, shape = [..., row].<br>When **axis** = -2 or input.shape.size() - 2, shape = [..., col].|
| otype  | Input      | Numeric type of the return value.<br>Currently supports int8 and uint8, corresponding to symmetric quantization and asymmetric quantization, respectively.|
| axis  | Input      | Axis along which quantization compression is performed.<br>Currently supports the last two axes, that is, -1/-2 or input.shape.size() - 1/input.shape.size() - 2.<br>**When input is 1D, only -1 is supported.** |
| zero_points  | Input      | Optional offset factor for asymmetric quantization.<br>Supported type: Tensor.<br>The tensor data type is the same as that of **input**, supporting **DT_FP32**.<br>Empty tensors are supported.<br>The shape has one fewer dimension than **input**, only supporting 1 to 3 dimensions.<br>The shape size must not exceed **2147483647** (that is, **INT32_MAX**).<br>When **axis** = -1 or input.shape.size() - 1, shape = [..., row].<br>When **axis** = -2 or input.shape.size() - 2, shape = [..., col]. |

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
x = pypto.tensor([3, 4], pypto.DT_FP32)
scale = pypto.tensor([3], pypto.DT_FP32)
zero_points = pypto.tensor([3], pypto.DT_FP32)

# fp32 -> int8 symmetric quantization.
y1 = pypto.quantize(x, scale, pypto.DT_INT8, -1, None)
# fp32 -> uint8 asymmetric quantization.
y2 = pypto.quantize(x, scale, pypto.DT_UINT8, -1, zero_points)
```

The results are as follows:

```python
Input  x:[[1.1, -2.2, 3.3, -4.4], [1.1, -2.2, 3.3, -4.4], [1.1, -2.2, 3.3, -4.4]]
Input  scale:[1.0, 1.0, 1.0]
Input zero_points:[-5.0, -5.0, -5.0]
Output y1:[[1, -2, 3, -4], [1, -2, 3, -4], [1, -2, 3, -4]]
Output y2:[[6, 3, 8, 1], [6, 3, 8, 1], [6, 3, 8, 1]]
```
