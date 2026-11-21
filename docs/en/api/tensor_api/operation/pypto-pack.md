# pypto.pack

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

Flattens the input tensor into a one-dimensional tensor and interprets the original bytes as uint8 elements.

## Prototype

```python
pack(self: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| self    | Input     | Source operand.<br>Supported data types: DT_BF16, DT_FP16, DT_FP32, DT_INT64, DT_UINT64, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor. Its shape is one-dimensional, the number of elements equals the number of bytes occupied by the input tensor elements, and each element is the uint8 representation of the corresponding input tensor element.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the input.

For example, if the input shape is [m, n], the output is [k], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([1, 2], pypto.DT_INT32)
y = pypto.pack(x)
```

Result example:

```python
Input data x: [[ 0x03040506 0x01050608]]
Output data y: [  6  5  4  3  8  6  5  1  ]
```
