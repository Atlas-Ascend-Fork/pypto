# pypto.acos

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:54:15.490Z pushedAt=2026-10-28T11:57:29.725Z nl=2 -->

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

Computes the arccosine of each element in the input tensor, operating element-wise. The result falls within the interval $[0, \pi]$. When the absolute value of an input element is greater than 1, the corresponding output is NaN.

$$
y_i = \arccos(x_i)
$$

## Prototype

```python
acos(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| input  | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of a tensor: DT_FP32, DT_FP16, DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**).|

## Return Value

Returns a tensor. Its shape is the same as that of the input tensor, its data type is the same as that of the input tensor, and its elements are the arccosine values of the corresponding elements of the input tensor.

## Constraints

1. Considering the input, output, and temporary space usage, the TileShape size must satisfy additional constraints. Assume that TileShape is \[a,b,c,d\], and let $d_{align}=CeilAlign(d, 64)$. Then the total UB space usage is:
   $$
   6*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.acos(x)
```

The results are as follows:

```python
Input data x: [-1.0000, 0.0000, 1.0000]
Output data y: [3.1416, 1.5708, 0.0000]
```
