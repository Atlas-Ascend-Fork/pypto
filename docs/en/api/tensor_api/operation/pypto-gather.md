# pypto.gather

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Not supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Not supported
<!-- end id8 -->
<!-- npu="9050" id9 -->
- Kirin 9050 processors: Not supported
<!-- end id9 -->

## Description

Extracts values from the input tensor along the dimension specified by dim at the given indices defined by index, and returns the result. For example, for a 3D tensor, the computation is as follows:

$$
\begin{cases}
output[i,j,k] = input[index[i,j,k], j, k] & \text{if } dim = 0; \\
output[i,j,k] = input[i, index[i,j,k], k] & \text{if } dim = 1; \\
output[i,j,k] = input[i,j, index[i,j,k]] & \text{if } dim = 2.
\end{cases}
$$

## Prototype

```python
gather(input: Tensor, dim: int, index: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports 1 to 5 dimensions, and the shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Dimension along which to index.<br>It supports any valid dimension index, ranging from -input.dim to input.dim - 1.|
| index   | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_INT32 and DT_INT64.<br>Empty tensors are not supported. The shape can have 1 to 4 dimensions. The shape size on each axis of index must not exceed the corresponding shape size of input, and the values must be valid indices, that is, they must not exceed the shape size of input on the dim axis.|

## Return Value

Returns an output tensor. Its data type is the same as that of the input. Its shape is the same as that of the index.

## Constraints

1. index.dim = input.dim, and index.shape\[i\] <= input.shape\[i\] (i != dim), with values being valid indices, which means they must not exceed input.shape[dim].

2. dim: -input.dim <= dim < input.dim;

3. input.shape along the dim axis cannot be tiled, requiring viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \), while the shape sizes of the remaining dimensions are not restricted.

4. The dimensions of TileShape are the same as those of index, which are used to tile the input and index. The dim axis of the input cannot be tiled, and the total size of all input and output TileShape values cannot exceed the size of the UB memory.

5. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT16, DT_INT32, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT16, DT_INT32, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT16, DT_INT32, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id6 -->

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input is [x, y, z], dim is 1, the input index is [m, t, p], the output is [m, t, p], where m <= x and p <= z, and TileShape is set to [m1, t1, p1], then m1, t1, and p1 are used to tile the m, t, and p axes, respectively. The y axis cannot be tiled and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)
index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
dim = 0
y = pypto.gather(x, dim, index)
```

Result example:

```python
Input data x: [[0,  1,  2,  3,  4],
             [5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14]]
     index: [[0, 1, 2, 0],
             [1, 2, 0, 1],
             [2, 2, 1, 0]]
Output data y: [[0,  6,  12, 3],
             [5,  11, 2,  8],
             [10, 11, 7,  3]]
```
