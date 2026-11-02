# pypto.cbrt

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:12:34.675Z pushedAt=2026-10-28T11:57:29.801Z nl=2 -->

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

Computes the cube root of each element in the input tensor element-wise and returns a tensor with the same shape as the input. The computation formula is:

$$
y_i = \sqrt[3]{x_i}
$$

## Prototype

```python
cbrt(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of a tensor: DT_FP32, DT_FP16, DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor, whose data type and shape are the same as those of **input**.

## Constraints

1. The input tensor and the output tensor must have the same type.
2. Considering the combined usage of input, output, and temporary storage space, the TileShape size must satisfy additional constraints. Assuming the TileShape is \[a,b,c,d\], let $d_{align}=CeilAlign(d, 32/sizeof(dt))$. The total UB space usage is:

   $$
   4*a*b*c*d_{align}*sizeof(dt) <= UB
   $$

   where $dt$ is the data type of the input tensor.

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
x = pypto.tensor([1.0, 2.0], pypto.DT_FP32)
y = pypto.cbrt(x)
```

The results are as follows:

```python
Input  x:[[8.0, -8.0]]
Output y:[[2.0, -2.0]]
```
