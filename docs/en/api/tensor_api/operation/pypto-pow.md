# pypto.pow

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
- Kirin 9030 processors: Not supported
<!-- end id10 -->
<!-- npu="x90" id11 -->
- Kirin X90 processors: Not supported
<!-- end id11 -->
<!-- npu="9050" id15 -->
- Kirin 9050 processors: Not supported
<!-- end id15 -->


## Description

Computes the other-th power of each element in the input tensor element-wise and returns a tensor with the same shape as the input.

## Prototype

```python
pow(input: Tensor, other: Union[Tensor, int, float], precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>The tensor supports the following data types: DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT8, DT_UINT8, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other   | Input     | Exponent.<br>Supported types: Tensor, int, or float.<br>The tensor supports the following data types: DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT8, DT_UINT8, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | Input     | Precision mode enumeration type, which is used to control the precision mode of exponent computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>The default value is HIGH_PRECISION (high-precision mode).|

## Return Value

Returns a tensor with the same shape as the input, where each element is the other-th power of the corresponding element in the input tensor.

If other is an int, the data type of the returned tensor is the same as that of the input.

If other is a float and the input tensor type is DT_INT32, DT_FP32 is returned. Otherwise, the data type of the returned tensor is the same as that of the input.

If other is a tensor, see the section on data type promotion for the data type of the returned tensor.

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
2. If both inputs are tensors and their input types are int8, uint8, or int16, the two input parameters must have the same data type.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
4. Temporary space usage notes: If TileShape is \[a,b,c,d\], the required temporary space size is a\*b\*c\*d\*sizeof\(DT_FP32\) when the output data type is DT_INT32, DT_INT8, DT_UINT8, or DT_INT16. In other cases, the required temporary space size is as follows:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: a\*b\*c\*d\*sizeof\(DT_FP32\)
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id9 -->
5. UB space usage of the input tensor: When the output data type of pow is DT_FP16, DT_BF16, or DT_FP32, both inputs (when the exponent is a scalar, the scalar exponent is padded to a DT_FP32 tensor with the same shape as the input) are converted to DT_FP32 and then copied to the UB. In the UB, the last dimension of the input tensor is 32-byte aligned and padded, and then the second-to-last dimension is row-padded. If TileShape is \[a,b,c,d\], and let $d_{align}=CeilAlign(d, 8), k=d_{align}/8, p=\lceil8/k\rceil, and c_{pad}=c+p-1$, then the actual UB space usage of the input tensor is as follows:
   <!-- npu="950" id12 -->
   - Ascend 950PR&950DT products: No row padding. The actual UB space occupied by each input tensor is $a*b*c*d_{align}*sizeof(DT\_FP32)$. The total space occupied by the two inputs is $2*a*b*c*d_{align}*sizeof(DT\_FP32)$.
   <!-- end id12 -->
   <!-- npu="A3" id13 -->
   - Atlas A3 products: The actual UB space occupied by each input tensor is $a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)$. The total space occupied by the two inputs is $2*a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)$. Considering the temporary space usage notes in item 4, the total UB space usage must satisfy $2*a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)+a*b*c*(2*d+3*d/8)*sizeof(DT\_FP32) <= UB$.
   <!-- end id13 -->
   <!-- npu="910b" id14 -->
   - Atlas A2 products: The actual UB space occupied by each input tensor is $a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)$. The total space occupied by the two inputs is $2*a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)$. Considering the temporary space usage notes in item 4, the total UB space usage must satisfy $2*a*b*c_{pad}*d_{align}*sizeof(DT\_FP32)+a*b*c*(2*d+3*d/8)*sizeof(DT\_FP32) <= UB$.
   <!-- end id14 -->

## Data Type Promotion Rules

We define float32 > float16 > bfloat16 > int32.

1. If both input parameters are of type int8, uint8, or int16, the output type is the same as the input type.
2. If one input parameter type is float16 and the other is bfloat16, the output data type is float32.
3. In other cases, the output type is the larger value of the input parameter types. For example, if the input types are float32 and float16, the output type is float32. For details, see the following table.

| Parameter Type   | float32    | float16    | bfloat16   | int32      |
|-------------|------------|------------|------------|------------|
| **float32**     | float32    | float32    | float32    | float32    |
| **float16**     | float32    | float16    | **float32**    | float16    |
| **bfloat16**    | float32    | **float32**    | bfloat16   | bfloat16   |
| **int32**       | float32    | float16    | bfloat16   | int32      |

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
x = pypto.tensor([2, 2], pypto.DT_FP32)
a = 2
b = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.pow(x, a)
z = pypto.pow(x, b)
```

Result example:

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
