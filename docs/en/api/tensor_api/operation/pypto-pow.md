# pypto.pow

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:16:10.532Z pushedAt=2026-10-28T11:57:30.037Z nl=2 -->

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

Computes the **other**-th power of each element in the input tensor element-wise, and returns a tensor with the same shape as the input.

## Prototype

```python
pow(input: Tensor, other: Union[Tensor, int, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| input | Input | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT8, DT_UINT8, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| other | Input | Exponent.<br>Supported types: Tensor, int, or float.<br>Supported data types of Tensor: DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT8, DT_UINT8, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| precision_type | Input | Precision mode enumeration type, which controls the precision mode of exponent computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>Defaults to HIGH_PRECISION (high-precision mode). |

## Return Value

Returns a tensor with the same shape as the input, where each element is the **other**-th power of the corresponding element in the input tensor.

When **other** is an int, the data type of the returned tensor is the same as that of the input.

When **other** is a float, if the input tensor type is DT_INT32, DT_FP32 is returned; otherwise, the data type of the returned tensor is the same as that of the input.

When **other** is a tensor, see the data type promotion section for the data type of the returned tensor.

## Constraints

1. Usage notes for precision_type:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: Supported
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: Not supported. The instruction mode `INTRINSIC` is used by default.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: Not supported. The instruction mode `INTRINSIC` is used by default.
   <!-- end id6 -->
2. When both inputs are tensors and their data types are int8, uint8, or int16, the two input parameters must have the same data type.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
4. Temporary space usage notes: If TileShape is \[a,b,c,d\], when the output data type is DT_INT32, DT_INT8, DT_UINT8, or DT_INT16, the required temporary space size is a\*b\*c\*d\*sizeof\(DT_FP32\); in other cases, the required temporary space size is
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: a\*b\*c\*d\*sizeof\(DT_FP32\)
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id9 -->

## Data Type Promotion Rules

We define float32 > float16 > bfloat16 > int32.

1. When both input parameters are of type int8, uint8, or int16, the output type is the same as the input type.
2. When one input parameter type is float16 and the other is bfloat16, the output data type is float32.
3. In other cases, the output type is the larger of the input parameter types. For example, if the inputs are float32 and float16, the output is float32. Refer to the following table.

| parameter type    | float32    | float16    | bfloat16   | int32      |
|-------------|------------|------------|------------|------------|
| **float32**     | float32    | float32    | float32    | float32    |
| **float16**     | float32    | float16    | **float32**    | float16    |
| **bfloat16**    | float32    | **float32**    | bfloat16   | bfloat16   |
| **int32**       | float32    | float16    | bfloat16   | int32      |

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
x = pypto.tensor([2, 2], pypto.DT_FP32)
a = 2
b = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.pow(x, a)
z = pypto.pow(x, b)
```

The results are as follows:

```python
Input data x: [[1.0  2.0], [-3.0  4.0]]
Input data b: [[2.0  2.0], [1.0   1.0]]
Output data y: [[1.0  4.0], [9.0  16.0]]
Output data z: [[1.0  4.0], [-3.0  4.0]]
```

### High-Precision Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.tensor([2, 2], pypto.DT_FP16)
out = pypto.pow(x, y, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.pow(x, y, pypto.PrecisionType.INTRINSIC)
```
