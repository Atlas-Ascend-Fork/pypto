# pypto.hypot

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:48:36.290Z pushedAt=2026-10-28T11:57:29.937Z nl=2 -->

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

Computes the square root of the sum of squares of **input** and **other** element-wise (that is, the length of the hypotenuse of a right triangle). The computation formula is as follows:

$$
res_i = \sqrt{input_i^2 + other_i^2}
$$

## Prototype

```python
hypot(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX). |
| other   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. When both **input** and **other** are tensors, their data types must be the same.
2. For BF16 and FP16 types, the internal computation may increase precision to avoid intermediate overflow.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
# Example: Compute the hypotenuse of two sets of right-angle sides.
# First set: (3, 4) -> 5
# Second set: (5, 12) -> 13
a = pypto.tensor([3.0, 5.0], pypto.DT_FP32)
b = pypto.tensor([4.0, 12.0], pypto.DT_FP32)
out = pypto.hypot(a, b)
```

The results are as follows:

```python
Input data a:   [3.0, 5.0]
Input data b:   [4.0, 12.0]
Output data out: [5.0, 13.0]
```
