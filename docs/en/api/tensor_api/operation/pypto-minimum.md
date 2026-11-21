# pypto.minimum

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
<!-- npu="9030" id10 -->
- Kirin 9030 processors: Supported
<!-- end id10 -->
<!-- npu="x90" id11 -->
- Kirin X90 processors: Supported
<!-- end id11 -->
<!-- npu="9050" id16 -->
- Kirin 9050 processors: Supported
<!-- end id16 -->

## Description

Computes the minimum value between the input and another input. 2D, 3D, or 4D tensors are supported.

## Precautions

- **SymbolicScalar parameters are not supported.** To compare SymbolicScalar values, use the [SymbolicScalar.min()](../symbolic/pypto-SymbolicScalar-min.md) method.
- At least one of the two parameters must be of the Tensor type.

## Prototype

```python
minimum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported types: int, float, Element, and Tensor.<br>When the value is of the int or float type, it is automatically converted to the Element type DT_INT32 or DT_FP32. To use other data types, construct the value through Element.<br>Supported Tensor and Element data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| other   | Input     | Source operand.<br>Supported types: int, float, Element, and Tensor.<br>When the value is of the int or float type, it is automatically converted to the Element type DT_INT32 or DT_FP32. To use other data types, construct the value through Element.<br>Supported Tensor and Element data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).<br>The type and data type must be the same as those of source operand 1.|

At least one of source operand 1 and source operand 2 must be a tensor.

## Return Value

If both source operands are tensors, the two tensors must meet the broadcasting rules. This API returns a tensor with the same shape as the broadcasted source operand 1 and source operand 2, with the same data type as the source operands. The elements of the returned tensor are the element-wise minimum value of the source operand 1 and source operand 2. When the source operands are tensors, both source operand 1 and source operand 2 support multi-axis broadcasting.

If one of the two source operands is a tensor, a tensor with the same shape as the input tensor is returned. The elements of the returned tensor are the element-wise minimum value of the source operand 1 and source operand 2.

## Constraints

1. When both inputs are tensors, the following data types are supported:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT32, DT_UINT32, DT_FP32, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_INT64, and DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id6 -->
   <!-- npu="9030" id12 -->
   - Kirin 9030 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id12 -->
   <!-- npu="x90" id13 -->
   - Kirin X90 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id13 -->
   <!-- npu="9050" id17 -->
   - Kirin 9050 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id17 -->
2. When one input is of the Tensor type and the other is of the Element type, the supported data types are as follows:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16, DT_INT64, and DT_UINT64
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id9 -->
   <!-- npu="9030" id14 -->
   - Kirin 9030 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id14 -->
   <!-- npu="x90" id15 -->
   - Kirin X90 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id15 -->
   <!-- npu="9050" id18 -->
   - Kirin 9050 processors: DT_INT32, DT_INT16, DT_FP16, and DT_FP32
   <!-- end id18 -->
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
a = pypto.tensor([3], pypto.DT_INT32)
b = pypto.tensor([3], pypto.DT_INT32)
out = pypto.minimum(a, b)
```

Result example:

```python
Input data a: [0, 2, 4]
Input data b: [3, 1, 3]
Output data out: [0, 1, 3]
```
