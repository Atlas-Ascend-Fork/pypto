# pypto.mul

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:06:38.597Z pushedAt=2026-10-28T11:57:30.008Z nl=2 -->

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

Computes the element-wise product, expressed as follows:

$$
res_{i} = input_{i} * other_{i}
$$

## Prototype

```python
mul(input: Tensor, other: Union[Tensor, float, int]) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: **Tensor**. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |
| other   | Input      | Source operand.<br>Supported types: **float**, **int**, and **Tensor**. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. When both **input** and **other** are tensors, their data types must be the same.
2. When **other** is a scalar, if **input** is of a floating-point type, the scalar supports integer types (automatically converted to floating-point); if **input** is of an integer type, the scalar does not support floating-point types (an error is reported).
3. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT32, DT_FP32, DT_INT16, DT_FP16, DT_BF16, DT_INT64, DT_UINT64.
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16.
   <!-- end id6 -->
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 3], pypto.DT_FP32)
z = pypto.mul(x, y)
```

The results are as follows:

```python
Input data x:  [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Input data y:  [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Output data z:  [[1.0 4.0 9.0],
             [1.0 4.0 9.0]]
```
