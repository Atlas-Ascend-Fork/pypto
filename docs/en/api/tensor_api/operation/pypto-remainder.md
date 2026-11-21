# pypto.remainder

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
<!-- npu="9050" id12 -->
- Kirin 9050 processors: Not supported
<!-- end id12 -->

## Description

Performs a remainder operation on each element of input and the element at the corresponding position in other. The computation formula is as follows:

$$
res_i = input_i - other_i * floor(input_i / other_i)
$$

## Prototype

```python
remainder(
    input: Union[Tensor, int, float],
    other: Union[Tensor, int, float],
    precision_type: PrecisionType = PrecisionType.HIGH_PRECISION
) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>Supported types: Tensor, int, and float.<br>The data types supported by tensors vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other  | Input     | Source operand.<br>Supported types: Tensor, int, and float.<br>The data types supported by tensors vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | Input     | Precision mode enumeration type, which is used to control the precision mode of the remainder operation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>The default value is HIGH_PRECISION (high-precision mode).|

## Return Value

Returns the output tensor. Its shape is the broadcast shape of input and other. Its data type is the same as that of the input tensor.

## Constraints

1. Currently, mixed-precision type inputs are not supported. This means when all inputs are tensors, their data types must be the same. When one input is a scalar, the tensor data type must be the corresponding integer type or floating-point type.
2. When the input is of the integer type, **other** must not contain 0. The result of integer remainder is determined by the chip type, which may be 0 or -1.
3. If the data type of the input tensor is DT_INT32 and the data value is out of the range [–2^24, 2^24], the precision is not guaranteed.
4. Usage notes for precision_type:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: Supported
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: Not supported. The instruction mode `INTRINSIC` is used by default.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: Not supported. The instruction mode `INTRINSIC` is used by default.
   <!-- end id6 -->
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
6. A tensor supports the following data types:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, and DT_INT64
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16
   <!-- end id9 -->
7. Due to the use of temporary memory, when setting TileShape, ensure that the total size of the input tile, output tile, and temporary space is less than the available UB. The last two dimensions of TileShape are denoted as H and W (H is 1 for one-dimensional TileShape), align is denoted as 256/sizeof(output), and W aligned to align is denoted as WAlign:
   - If both input and other are tensors, the size of the temporary space is `(WAlign + CeilAlign(WAlign / 8, 256) / sizeof(output) + 256) * sizeof(output)` bytes.
   - If either input or other is not a tensor, the size of the temporary space is `(H + 1) * (WAlign + CeilAlign(WAlign / 8, 256) / sizeof(output) + 256) * sizeof(output)` bytes.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

As shown in the following non-broadcast example, the shapes of input and other are both [n], the output is [n], and the TileShape is set to [n1]. In this case, n1 is used to tile the n axis.

```python
pypto.set_vec_tile_shapes(16)
```

### API Call Example

The following is the API call snippet in the computational graph construction context. The first parameter of pypto.tensor is Shape, which does not initialize the example value. The specific input data is provided by the caller during execution. The high-precision mode and instruction mode use the same input shape.

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.remainder(a, b)
```

When the caller provides the following input data, the result is as follows:

```python
Input data a:    [7.0, 8.0, 9.0]
Input data b:    [-3.0, -3.0, -3.0]
Output data out:  [-2.0, -1.0, 0.0]
```

### High-Precision Mode Example

```python
a = pypto.tensor([3], pypto.DT_FP16)
b = pypto.tensor([3], pypto.DT_FP16)
out = pypto.remainder(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.remainder(a, b, pypto.PrecisionType.INTRINSIC)
```
