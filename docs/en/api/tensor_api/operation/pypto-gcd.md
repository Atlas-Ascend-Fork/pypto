# pypto.gcd

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:44:23.311Z pushedAt=2026-10-28T11:57:29.921Z nl=2 -->

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

Computes the greatest common divisor of the elements of **input** and **other**.

## Prototype

```python
gcd(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                                                   |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT8, DT_INT16, DT_INT32, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports multi-dimensional broadcasting to the same shape. The shape size must not exceed 2147483647 (INT32_MAX).           |
| other  | Input      | Source operand.<br>Supported types: int and Tensor.<br>Supported data types of Tensor: DT_INT8, DT_INT16, DT_INT32, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions and supports multi-dimensional broadcasting to the same shape. The shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the broadcast shape of input and other.

## Constraints

1. **input** and **other** have the same data type.
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through set_vec_tile_shapes. The last axis of the TileShape must be 32-byte-aligned.

The TileShape must have the same number of dimensions as the output.

Example 1: In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)
y = pypto.tensor([2, 3], pypto.DT_INT32)
z = pypto.gcd(x, y)
# Using a scalar
c = pypto.gcd(x, 2)
```

The results are as follows:

```python
Input data x: : [[9 9 9],
             [6 6 6]]
Input data y:   [[1 2 3],
             [1 2 3]]
Output data z:   [[1 1 3],
             [1 2 3]]
Output data c:   [[1 1 1],
             [2 2 2]]
```
