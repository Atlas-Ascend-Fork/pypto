# pypto.acosh

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:55:09.572Z pushedAt=2026-10-28T11:57:29.728Z nl=2 -->

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

Computes the inverse hyperbolic cosine of each element of the input tensor.

$$
y_i = \operatorname{acosh}(x_i) = \ln(x_i + \sqrt{x_i^2 - 1})
$$

## Prototype

```python
acosh(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| input  | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of a tensor: DT_FP32, DT_FP16, DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**).<br>The mathematical domain requires that input elements be greater than or equal to 1. |

## Return Value

Returns the output tensor, whose **Shape** is the same as that of `input`, whose data type is the same as that of `input`, and whose element values are the inverse hyperbolic cosine values of the corresponding elements of the input tensor.

## Constraints

1. Considering the input, output, and temporary space usage, the TileShape size has additional constraints. Assume that TileShape is \[a,b,c,d\], and let $d_{align}=CeilAlign(d, 8)$. Then the total UB space usage is:

   $$
   5*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

The TileShape must have the same number of dimensions as the output.

If the input `input` shape is `[m, n]` and the output is `[m, n]`, with TileShape set to `[m1, n1]`, then `m1` and `n1` are used to split the `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.acosh(x)
```

The results are as follows:

```python
Input data x: [1.0000, 2.0000, 3.0000, 4.0000]
Output data y: [0.0000, 1.3170, 1.7627, 2.0634]
```
