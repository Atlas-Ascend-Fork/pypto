# pypto.scatter\_update

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:29:40.666Z pushedAt=2026-10-28T11:57:30.088Z nl=2 -->

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

Function 1: An in-place operation that updates the 4D **src** to the 4D **input** based on the 2D index **index**. The computation formula is as follows:

$$
input\left[\frac{\text{index}[i][j]}{\text{blockSize}}\right]\left[\text{index}[i][j] \% \text{blockSize}\right][0][\dots] = src[i][j][0][\dots]
$$

Function 2: An in-place operation that updates the 2D **src** to the 2D **input** based on the 2D **index**. The computation formula is as follows (where s is the size of the second dimension of **index**, that is, **index.shape[1]**):

$$
input[[\text{index}[i][j]][\dots]] = src[i*s + j][\dots]
$$

## Prototype

```python
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Supported type is: Tensor.<br>Supported data type of Tensor is: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8.<br>Supported dimensions: 2D, 4D.<br>2D shape [blockNum * blockSize, d], 4D shape [blockNum, blockSize, 1, d].<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| dim     | Input      | Keep the default value -2. |
| index   | Input      | A set of indices of **input**.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_INT64, DT_INT32, and DT_INT16.<br>Supported dimensions: 2D.<br>Shape [b, s]. |
| src     | Input      | src is a set of update values.<br>Supported type is: Tensor.<br>Supported data type of Tensor is: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8. The data type must be the same as that of input.<br>Supported dimensions: 2D, 4D.<br>2D shape [b * s, d], 4D shape [b, s, 1, d].<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |

## Return Value

Returns the updated input. This is an inplace operation, and the original values of the target buffer are retained at non-indexed positions. If different formal parameters are used for input and output, the same actual buffer must be bound when calling.

## Constraints

Broadcast constraint: broadcast is not supported.

Index constraint: When index contains duplicate indices pointing to the same target position, the write order and the final written result are not guaranteed to be deterministic.

Tensor format constraint: Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

ViewShape constraint: In 2D scenarios, the ViewShape is \[viewB \* s, d\]; in 4D scenarios, the ViewShape is \[viewB, viewS, 1, d\]. The last axis d is not tiled. In 2D scenarios, \[viewB \* s, d\] performs tiling on src, whose dimension 0 is a multiple of s, the 1st dimension of index; [viewB, s] performs tiling on index. In 4D scenarios, \[viewB, viewS, 1, d\] performs tiling on src, and \[viewB, viewS\] performs tiling on index.

TileShape constraint: In 2D scenarios, TileShape is \[tileSize, d\]; in 4D scenarios, TileShape is \[tileB, tileS, 1, d\]. The last axis d is not split. Let index.shape\[1\] be s. In 2D scenarios, when TileShape\[0\] is not greater than s, TileShape\[0\] must be a divisor of s; when TileShape\[0\] is greater than s, TileShape\[0\] must be a multiple of s. For example, when src is \[12, 64\] and index is \[3, 4\], TileShape can be \[1, 64\], \[2, 64\], \[4, 64\], \[8, 64\], or \[12, 64\]. In 4D scenarios, TileShape performs splitting on src, and \[tileB, tileS\] performs splitting on index. Since TileShape splitting targets src and index, the sum of the tile sizes must be less than the UB limit.

2D example: input: [16, 8], index: [5, 2], src: [10, 8], viewShape: [viewB \* s, 8]. viewB must be an integer, that is, dimension 0 must be a multiple of s. When TileShape\[0\] is not greater than s, it can be 1 or 2; when TileShape\[0\] is greater than s, it must be a multiple of s, such as 4, 6, 8, or 10.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape dimensions must be consistent with the input src.

Both the input and the output reside in gm and do not involve tiling. The input index and the input src must be moved into ub, which involves tiling.

For example, if the input is [t, d], dim is -2, the input index is [b, s], the input src is [bs, d], where bs=b*s, and the output is [t, d], with the TileShape set to [bs1, d1], then bs1 is used to tile the bs axis, the d axis is not allowed to be tiled, and d1 must be equal to d.

```python
pypto.set_vec_tile_shapes(16, 64)
```

### API Call Example

- Update the 2D input with the 2D src according to the 2D index. The following uses move to write the update result back to the same target tensor:

    ```python
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(4, 3)
    x.move(pypto.scatter_update(x, -2, y, z))
    ```

    The following input data must be provided by the caller. `pypto.tensor` only declares the tensor and does not initialize it to 0 by default. Non-indexed positions in the result example retain the input values:

    ```python
    Input data x:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12],
               [13 14 15],
               [16 17 18],
               [19 20 21],
               [22 23 24]]
    Input data y:[[1 2],
               [4 5]]
    Input data z:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12]]
    Output data x:[[1 2 3],
               [1 2 3],
               [4 5 6],
               [10 11 12],
               [7 8 9],
               [10 11 12],
               [19 20 21],
               [22 23 24]]
    ```

- Update the 4D input with the 4D src according to the 2D index. The following uses move to write the update result back to the same target tensor:

    ```python
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(1, 2, 1, 3)
    x.move(pypto.scatter_update(x, -2, y, z))
    ```

    The following input data must be provided by the caller. `pypto.tensor` only declares the tensor and does not initialize it to 0 by default. Non-indexed positions in the result example retain the input values:

    ```python
    Input data x:[[
                 [[1 2 3]],
                 [[4 5 6]],
                 [[7 8 9]],
                 [[10 11 12]],
                 [[13 14 15]],
                 [[16 17 18]],
               ],
               [
                 [[19 20 21]],
                 [[22 23 24]],
                 [[25 26 27]],
                 [[28 29 30]],
                 [[31 32 33]],
                 [[34 35 36]],
               ]]
    Input data y:[[1 8],
               [4 10]]
    Input data z:[[
                 [[1 2 3]],
                 [[4 5 6]],
               ],
               [
                 [[7 8 9]],
                 [[10 11 12]],
               ]]
    Output data x:[[
                 [[1 2 3]],
                 [[1 2 3]],
                 [[7 8 9]],
                 [[10 11 12]],
                 [[7 8 9]],
                 [[16 17 18]],
               ],
               [
                 [[19 20 21]],
                 [[22 23 24]],
                 [[4 5 6]],
                 [[28 29 30]],
                 [[10 11 12]],
                 [[34 35 36]],
               ]]
    ```
