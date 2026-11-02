# pypto.scatter\_

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:27:17.544Z pushedAt=2026-10-28T11:57:30.081Z nl=2 -->

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

Writes the value of **src** into **input**. The write position is specified by **index**. The 3D computation formula is as follows, and other dimensions follow the same pattern:
<br>When **src** is a fixed scalar:
$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src & \text{if } dim = 2
\end{cases}
$
<br>When **src** is a tensor:
$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 2
\end{cases}
$

## Prototype

```python
scatter_(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input   | Input      | Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, and DT_INT64.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| dim     | Input      | Specifies the dimension used for indexing, supporting any dimension within the range of **input** dimensions.<br>Valid dimension index, ranging from -input.dim to input.dim - 1. |
| index   | Input      | A set of indices of **input**.<br>Supported type: Tensor.<br>Supported data types of Tensor: INT64 and INT32.<br>Supported dimensions: consistent with those of **input**.<br>For all dimensions where d != dim, the following requirement must be met: index.size(d) <= input.size(d).<br>When **src** is a tensor, all dimensions must meet: index.size(d) <= src.size(d).<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| src     | Input      | src is the scalar or tensor used for updating.<br>When src is an Element, supported data types are: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, and DT_INT64. INF/NAN input is not supported.<br>When src is a tensor, supported data types are: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, and DT_INT64. The data type must be the same as that of input.<br> |
| reduce  | Input      | The reduction operation to apply, supporting 'add' or 'multiply'. If not specified, it defaults to direct replacement. |

## Return Value

Returns the updated **input**. This is an in-place operation.

## Constraints

1. Broadcast constraint: **input** and **index** do not support broadcast.

2. The `dim` axis of `input.shape` cannot be tiled. The dimensions of `viewshape` are the same as those of **input**, requiring `viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \)`. The shape sizes of the remaining dimensions are not restricted.

3. The dim axis of `input.shape` cannot be tiled. The dimensions of `tileshape` are the same as those of **input**, `tileshape\[dim\] \>= viewshape\[dim\]`. The shape sizes of the remaining dimensions are not restricted. **input**, **index**, and the result are all placed in UB, and the total `tileshape` size of all inputs and outputs must not exceed the UB memory size.

4. When splitting the non-dim axes of `input.shape` and `index.shape`, after splitting `viewshape[non dim]`, the number of tile blocks on the non-dim axes of **input** and **index** must be the same. When splitting `tileshape`, the number of tile blocks on the non-dim axes of **input** and **index** must also be the same.

5. When **src** is a tensor, **reduce** is None, and **index** contains non-unique indices pointing to the same position, the behavior is undefined, and an arbitrary value from **src** will be selected and written.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input **input** has shape [a, b, c], **dim** is 1, **index** is [m, t, p] (where m<=a and p<=c), **src** is [x, y, z] (where x>=m, y>=t, and z>=p), and the output is [a, b, c], set **TileShape** to [m1, t1, p1]. Then m1 and p1 are used to split the m and p axes, respectively. t1 must be greater than or equal to both b and t. The axis corresponding to **dim** cannot be split, and the b axis and t axis must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

- Update the values at the corresponding indices of a 2D **input** based on a 2D **index**.

    ```python
    x = pypto.tensor([3, 5], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    o = pypto.scatter_(x, 0, y, 2.0)
    ```

    The results are as follows:

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
