# pypto.neg

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Supported
<!-- end id8 -->
<!-- npu="9050" id11 -->
- Kirin 9050 processors: Supported
<!-- end id11 -->

## Description

Computes the negative of each element in the input tensor element-wise, and returns a tensor with the same shape as the input.

## Prototype

```python
neg(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>The data types supported by tensors vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor with the same shape and data type as the input tensor. Each element of the output tensor is the negative value of the corresponding element in the input tensor.

## Constraints

1. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, and DT_INT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16
   <!-- end id6 -->
   <!-- npu="9030" id9 -->
   - Kirin 9030 processors: DT_INT16, DT_INT32, DT_FP16, and DT_FP32
   <!-- end id9 -->
   <!-- npu="x90" id10 -->
   - Kirin X90 processors: DT_INT16, DT_INT32, DT_FP16, and DT_FP32
   <!-- end id10 -->
   <!-- npu="9050" id12 -->
   - Kirin 9050 processors: DT_INT16, DT_INT32, DT_FP16, and DT_FP32
   <!-- end id12 -->
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.neg(x)
```

Result example:

```python
Input data x: [1.0, 2.0, 3.0, 4.0, 5.0]
Output data y: [-1.0, -2.0, -3.0, -4.0, -5.0]
```
