# pypto.concat

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:21:14.384Z pushedAt=2026-10-28T11:57:29.833Z nl=2 -->

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

Concatenates multiple input tensors along the specified dimension (**dim**) and returns a concatenated tensor.

## Prototype

```python
concat(tensors: List[Tensor], dim: int = 0) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
| --------- | ------------ | ----------- |
| tensors | Input | Source operands. Supported type: Tensor. Supported data types of Tensor: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16, DT_INT64, and DT_UINT64. Empty tensors are not supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| dim | Input | Dimension along which concatenation is performed. Supported data type: **int**, defaulting to **0**. |

## Return Value

Returns the output tensor. The data type and shape of the tensor are the same as those of any tensor in **tensors** (except for the dimension corresponding to **dim**). The dimension corresponding to **dim** is the sum of the corresponding dimensions of all tensors in **tensors**.

## Constraints

1. The size of the source operand tensors must be greater than or equal to 2, that is, len(tensors)>=2, and less than or equal to 128. (Input of a single tensor is supported, but its precision is not guaranteed for the time being.)

2. The input tensors must have the same data type and the same number of dimensions. In addition, every dimension value except the dimension to be concatenated (**dim**) must be the same. For the dimension to be concatenated, **validShape** must equal the corresponding dimension value of the tensor, and for the remaining dimensions, the **validShape** of all tensors must be the same.

3. dim: -input.dim <= dim < input.dim (input corresponds to any tensor in tensors).

4. When **viewshape** is set, the dimension corresponding to **dim** is not tiled (that is, the value of **viewshape** is greater than or equal to that of any tensor in **tensors**).

5. The valid shape of the output tensor must be correctly specified prior to calling **concat**; this API does not perform automatic inference.

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

If the input tensors have dimensions **[m, c1, p]** and **[m, c2, p]**, the output is **[m, c1+c2, p]**, and **TileShape** is set to **[m1, n1, p1]**, **m1** and **p1** are used to tile the **m** and **p** axes respectively, and **n1** is used to tile the **c1** and **c2** axes.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
a = pypto.ones([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 1s
b = pypto.zeros([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 0s
out = pypto.concat([a, b], dim = 0)
```

The results are as follows:

```python
Input data a:   [[1.0 1.0],
              [1.0 1.0]]
Input data b:   [[0.0 0.0],
              [0.0 0.0]]
Output data out: [[1.0 1.0],
              [1.0 1.0],
              [0.0 0.0],
              [0.0 0.0]]

```
