# pypto.var

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:47:18.600Z pushedAt=2026-10-28T11:57:30.155Z nl=2 -->

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

Computes the variance of all data along the **dim** dimension of the input tensor. The computation formula is:
$
\sigma^2 = \frac{1}{\max(0, ~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2
$

## Prototype

```python
var(input: Tensor, dim: Union[int, List[int], Tuple[int]] = None, *, correction: float = 1, keepdim: bool = False)
```

## Parameters

| Parameter | Input/Output | Description |
|------------|---------- |----------------------------------------------------------------------|
| input      | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of a tensor: DT_FP32, DT_FP16, DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |
| dim        | Input      | Dimension along which reduction is performed.<br>Any single axis or multiple axes are supported.<br>Defaults to **None**, which means all axes. |
| correction | Input      | Correction coefficient.<br>Difference between the sample size and the sample degrees of freedom.<br>Defaults to Bessel's correction, that is, correction=1. |
| keepdim    | Input      | Whether to retain the reduced dimension after reduction, defaulting to **False**. |

## Return Value

Returns a tensor. Its data type is the same as that of the input tensor.

When **keepdim** is **True**, the shape of the corresponding **dim** is reduced to 1 while the shapes of other axes remain unchanged; when **keepdim** is **False**, the corresponding **dim** is removed.

## Constraints

1. The **dim** axis of **input.shape** cannot be tiled. The dimension count of **viewshape** is the same as that of **input**, requiring viewshape\[dim\] \== input.shape\[dim\], and the shape sizes of the remaining dimensions are not restricted.
2. Duplicate values are not supported in **dim**, and len(dim) <= input.dim.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same dimension count as the input.

Example 1: If the input shape is [m, n] and the output is [m, 1], set the TileShape to [m1, n1], where m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.var(x, 1, correction=1, keepdim=True)
```

The results are as follows:

```txt
Input data x: [[1., 2., 3.],
            [4., 5., 6.]]
Output data y: [[1.],
            [1.]]
```
