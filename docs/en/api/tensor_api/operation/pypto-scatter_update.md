# pypto.scatter\_update

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
<!-- npu="9050" id9 -->
- Kirin 9050 processors: Not supported
<!-- end id9 -->


## Description

Function 1: An in-place operation that updates the 4D src to the 4D input based on the 2D index. The computation formula is as follows:

$$
input\left[\frac{\text{index}[i][j]}{\text{blockSize}}\right]\left[\text{index}[i][j] \% \text{blockSize}\right][0][\dots] = src[i][j][0][\dots]
$$

Function 2: An in-place operation that updates the 2D src to the 2D input based on the 2D index. The computation formula is as follows (where s is the size of the second dimension of index, that is, index.shape[1]):

$$
input[[\text{index}[i][j]][\dots]] = src[i*s + j][\dots]
$$

## Prototype

```python
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Supported type: Tensor.<br>The supported data types vary by model. For details, see [Constraints](#constraints).<br>Supported dimensions: 2D and 4D.<br>2D shape: [blockNum * blockSize, d]; 4D shape: [blockNum, blockSize, 1, d]<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Retain the default value -2.|
| index   | Input     | A set of indices of input.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_INT64, DT_INT32, and DT_INT16.<br>Supported dimensions: 2D<br>Shape[b, s] |
| src     | Input     | A set of update values.<br>Supported type: Tensor.<br>The supported data types vary by model. For details, see [Constraints](#constraints). The data type is the same as that of input.<br>Supported dimensions: 2D and 4D.<br>2D shape: [b * s, d]; 4D shape: [b, s, 1, d]<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the updated input. This is an inplace operation. The original values of the target buffer are retained at the non-indexed positions. If the input and output use different parameters, the same actual buffer must be bound during API calls.

## Constraints

1. Broadcast constraints: Broadcast is not supported.
2. Index constraints: When index contains duplicate indices pointing to the same target position, the write sequence and the final written result are not guaranteed to be deterministic.
3. Tensor format constraints: Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
4. ViewShape constraints: In 2D scenarios, ViewShape is \[viewB \* s, d\]. In 4D scenarios, ViewShape is \[viewB, viewS, 1, d\]. The last axis d is not tiled. In 2D scenarios, \[viewB \* s, d\] tiles src, whose dimension 0 is a multiple of s, the 1st dimension of index. \[viewB, s\] tiles index. In 4D scenarios, \[viewB, viewS, 1, d\] tiles src, and \[viewB, viewS\] tiles index.
5. TileShape constraints:
   - In 2D scenarios, TileShape is \[tileSize, tileD\], and index.shape\[1\] is set to s. When TileShape\[0\] is not greater than s, it must be a divisor of s. When TileShape\[0\] is greater than s, it must be a multiple of s. For example, if src is \[12, 64\] and index is \[3, 4\], TileShape can be \[2, 64\], \[4, 64\], or \[12, 64\]. When tileD is less than d, \[4, 32\] may also be used.
   - In 4D scenarios, TileShape is \[tileB, tileS, 1, tileD\].
6. Both the input and output reside in gm and do not occupy the UB space. The input index and input src need to be tiled based on TileShape and then moved to the UB. The sum of the tile sizes of the two inputs must not exceed the UB limit.
7. A tensor supports the following data types:
   <!-- npu="950" id6 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_FP8E4M3, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id6 -->
   <!-- npu="A3" id7 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id7 -->
   <!-- npu="910b" id8 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id8 -->

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the input src.

If the input is [blockNum \* blockSize, d], dim is -2, the input index is [b, s], the input src is [b \* s, d], the output is [blockNum \* blockSize, d], and TileShape is set to [tileSize, tileD], then tileSize is used to tile the bs axis, and tileD is used to tile the d axis.

If the input is [blockNum, blockSize, 1, d], dim is -2, the input index is [b, s], the input src is [b, s, 1, d], the output is [blockNum, blockSize, 1, d], and TileShape is set to [tileB, tileS, 1, tileD], then tileB is used to tile the b axis, tileS is used to tile the s axis, and tileD is used to tile the d axis.

```python
pypto.set_vec_tile_shapes(16, 64)
```

### API Call Example

- Update the 2D src to the 2D input based on the 2D index. The following uses move to write the update result back to the same target tensor:

    ```python
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(4, 3)
    x.move(pypto.scatter_update(x, -2, y, z))
    ```

    The following input data needs to be provided by the caller. `pypto.tensor` only declares the tensor and does not initialize it to 0 by default. Non-indexed positions in the result example retain the input values:

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

- Update the 4D src to the 4D input based on the 2D index. The following uses move to write the update result back to the same target tensor:

    ```python
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    pypto.set_vec_tile_shapes(1, 2, 1, 3)
    x.move(pypto.scatter_update(x, -2, y, z))
    ```

    The following input data needs to be provided by the caller. `pypto.tensor` only declares the tensor and does not initialize it to 0 by default. Non-indexed positions in the result example retain the input values:

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
