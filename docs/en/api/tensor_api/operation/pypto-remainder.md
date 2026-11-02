# pypto.remainder

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:22:26.077Z pushedAt=2026-10-28T11:57:30.060Z nl=2 -->

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

Performs a remainder operation on each element of **input** and the element at the corresponding position in **other**. The computation formula is as follows:

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

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported types: Tensor, int, float.<br>The data types supported by Tensor vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions, and multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| other  | Input      | Source operand.<br>Supported types: Tensor, int, float.<br>The data types supported by Tensor vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions, and multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| precision_type | Input      | Precision mode enumeration, used to control the precision mode of the remainder computation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br>Defaults to **HIGH_PRECISION** (high-precision mode). |

## Return Value

Returns the output tensor, whose shape is the size after broadcasting **input** and **other**, and whose data type is the same as that of the input tensor.

## Constraints

1. Mixed-precision type inputs are not supported. That is, when all inputs are tensors, their data types must be the same. When one input is a scalar, the tensor data type must be the corresponding integer type (DT_INT32, DT_INT16, or DT_INT64, where DT_INT64 is supported only by Ascend 950PR&950DT products) or floating-point type (DT_FP32, DT_FP16, DT_BF16);
2. When **input** is of an integer data type, **other** must not contain 0. The result of integer remainder is determined by the chip type and may be 0 or -1.
3. If the data type of the input tensor is **DT_INT32** and the data range exceeds \[-2^24, 2^24\], precision is not guaranteed.
4. Usage notes for precision_type:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: Supported
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: Not supported. The `INTRINSIC` instruction mode is used by default.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: Not supported. The `INTRINSIC` instruction mode is used by default.
   <!-- end id6 -->
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
6. Tensor data type description:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT64.
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16.
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16.
   <!-- end id9 -->
7. Because temporary memory is used, when setting TileShape, ensure that the total occupancy of the input tile, output tile, and temporary space is less than the available UB. Let the last two dimensions of TileShape be H and W (for a one-dimensional TileShape, H is 1), and let align be 256/sizeof\(output\). W aligned to align is denoted as WAlign:
   - If both input and other are tensors, the temporary space size is `(WAlign + CeilAlign(WAlign / 8, 256) / sizeof(output) + 256) * sizeof(output)` bytes.
   - If either input or other is not a tensor, the temporary space size is `(H + 1) * (WAlign + CeilAlign(WAlign / 8, 256) / sizeof(output) + 256) * sizeof(output)` bytes.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through set_vec_tile_shapes.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b)
```

The results are as follows:

```python
Input data a:    [7.0, 8.0, 9.0]
Input data b:    [-3.0, -3.0, -3.0]
Output data out:  [-2.0, -1.0, 0.0]
```

### High-Precision Mode Example

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP16)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP16)
out = pypto.remainder(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b, pypto.PrecisionType.INTRINSIC)
```
