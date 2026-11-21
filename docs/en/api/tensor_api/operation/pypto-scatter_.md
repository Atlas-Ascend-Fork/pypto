# pypto.scatter\_

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

Writes the value of src to input. The write position is specified by index. The 3D computation formula is as follows, and other dimensions follow the same pattern:
<br>When src is a fixed scalar:
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src & \text{if } dim = 2
\end{cases}
$$
<br>When src is a tensor:
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 2
\end{cases}
$$

## Prototype

```python
scatter_(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Supported type: Tensor.<br>The tensor supports the following data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, and DT_INT64.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Dimension used for indexing. Any dimension within the input dimension range is supported.<br>It supports any valid dimension index, ranging from -input.dim to input.dim - 1.|
| index   | Input     | A set of indices of input.<br>Supported data type: Tensor.<br>Supported data types of Tensor: INT64 and INT32.<br>Supported dimensions: same as those of input.<br>For all dimensions where d != dim, index.size(d) <= input.size(d) must be satisfied.<br>When src is a tensor, all dimensions must satisfy index.size(d) <= src.size(d).<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| src     | Input     | Scalar or tensor used for updating.<br>When src is an element, the supported data types are DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, and DT_INT64. The input cannot be INF or NAN.<br>When src is a tensor, the supported data types are DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, and DT_INT64. The data type must be the same as that of input.<br>When src is a tensor, empty tensors are not supported.<br>|
| reduce  | Input     | Reduction operation to be applied. The value can be 'add' or 'multiply'. If not specified, it defaults to direct replacement.|

## Return Value

Returns the updated input, which is an inplace operation.

## Constraints

1. Broadcasting restriction: input and index do not support broadcasting.

2. input.shape along the dim axis cannot be tiled. The dimensions of viewshape must be the same as those of input, requiring viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \). The shape sizes of the remaining dimensions are not restricted.

3. input.shape along the dim axis cannot be tiled. The dimensions of TileShape must be the same as those of input, requiring TileShape\[dim\] \>= viewshape\[dim\]. The shape sizes of the remaining dimensions are not restricted. The input index and result are stored in the UB. The total size of all TileShape values for all inputs and outputs must not exceed the UB memory capacity.

4. For the non-dim axis tiling of input.shape and index.shape, the number of non-dim axis tiles for input and index must be the same after viewshape[non dim] is tiled. During TileShape tiling, ensure that the number of non-dim axis tiles for input and index is the same.

5. If src is a tensor, reduce is None, and index contains non-unique indices pointing to the same position, the behavior is undefined, and an arbitrary value from src will be selected and written.

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input shape is `[a, b, c]`, dim is 1, index is `[m, t, p]`(where m<=a, p<=c), src is `[x, y, z]`(where x>=m, y>=t, z>=p), the output is `[a, b, c]`, and TileShape is set to `[m1, t1, p1]`, then m1 and p1 are used to tile the m and p axes, respectively. t1 must be greater than or equal to b and t. The axis corresponding to dim cannot be tiled, and the b and t axes must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

- Update the values at the corresponding indices of a 2D input based on a 2D index.

    ```python
    x = pypto.tensor([3, 5], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    o = pypto.scatter_(x, 0, y, 2.0)
    ```

    Result example:

    ```txt
    Input data x:[[0 0 0 0 0],
               [0 0 0 0 0],
               [0 0 0 0 0]]
    Input data y:[[1 2],
               [0 1]]
    Output data o:[[2.0 0   0 0 0],
               [2.0 2.0 0 0 0],
               [0   2.0 0 0 0]]
    ```
