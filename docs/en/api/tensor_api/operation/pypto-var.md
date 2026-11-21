# pypto.var

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Computes the variance of all data in the dim dimension of the input tensor. The computation formula is as follows:
$$
\sigma^2 = \frac{1}{\max(0, ~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2
$$

## Prototype

```python
var(input: Tensor, dim: Union[int, List[int], Tuple[int]] = None, *, correction: float = 1, keepdim: bool = False)
```

## Parameters

| Parameter     | Input/Output| Description                                                                |
|------------|---------- |----------------------------------------------------------------------|
| input      | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim        | Input     | Dimension for reduction.<br>Any single axis or multiple axes are supported.<br>The default value is None, indicating all axes.|
| correction | Input     | Correction coefficient.<br>Difference between the sample size and the sample degrees of freedom.<br>By default, Bessel correction is used, that is, correction = 1.|
| keepdim    | Input     | Whether to retain the reduced dimension after reduction. The default value is False.|

## Return Value

Returns a tensor. Its data type is the same as that of the input tensor.

If keepdim is True, the shape of the corresponding dim is reduced to 1, and the shapes of other axes remain unchanged. If keepdim is False, the corresponding dim is removed.

## Constraints

1. input.shape along the dim axis cannot be tiled. The dimensions of viewshape must be the same as those of input, requiring viewshape\[dim\] \== input.shape\[dim\]. The shape sizes of the remaining dimensions are not restricted.
2. dim does not support duplicate values, and len(dim) <= input.dim.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the input.

Example 1: If the input shape is [m, n], the output is [m, 1], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.var(x, 1, correction=1, keepdim=True)
```

Result example:

```txt
Input data x: [[1., 2., 3.],
            [4., 5., 6.]]
Output data y: [[1.],
            [1.]]
```
