# pypto.add

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

Adds the input and other values, performing element-wise operations. The computation formula is as follows:

$$
res_i = input_i + other_i
$$

## Prototype

```python
add(input: Tensor, other: Union[Tensor, float, int]) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other   | Input     | Source operand.<br>Supported types: float, int, and Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type is the same as that of input and other, and its shape is the broadcast shape of input and other.

## Constraints

1. If both input and other are tensors, their data types must be the same.
2. When other is a scalar, if input is of the floating-point type, the scalar supports the integer type (automatically converted to the floating-point type); if input is of the integer type, the scalar does not support the floating-point type (an error is reported).
3. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_INT64, and DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id6 -->
   <!-- npu="9030" id9 -->
   - Kirin 9030 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id9 -->
   <!-- npu="x90" id10 -->
   - Kirin X90 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id10 -->
   <!-- npu="9050" id12 -->
   - Kirin 9050 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id12 -->
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], other is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], other is [m, 1], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.add(a, b)
```

Result example:

```python
Input data a:   [[1.0 2.0 3.0]]
Input data b:   [[2.0 3.0 4.0]]
Output data out: [[3.0 5.0 7.0]]
```
