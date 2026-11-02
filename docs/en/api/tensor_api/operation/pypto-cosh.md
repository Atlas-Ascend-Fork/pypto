# pypto.cosh

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:21:07.683Z pushedAt=2026-10-28T11:57:29.830Z nl=2 -->

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

Computes the hyperbolic cosine of each element in the input tensor, performing element-wise operations.

$$
y_i = \cosh(x_i) = \frac{e^{x_i} + e^{-x_i}}{2}
$$

## Prototype

```python
cosh(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| input  | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of a tensor: DT_FP32, DT_FP16, DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**).|

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the hyperbolic cosine values of the corresponding elements of the input tensor.

## Constraints

1. Considering the total input, output, and temporary storage space, the TileShape size must satisfy additional constraints. Assuming the TileShape is \[a,b,c,d\], then

    $$
    3*a*b*c*CeilAlign(d, 8)*sizeof(DT\_FP32) <= UB
    $$.
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
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.cosh(x)
```

The results are as follows:

```python
Input data x: [0.0000, 1.0000, 2.0000, -1.0000]
Output data y: [1.0000, 1.5431, 3.7622, 1.5431]
```
