# pypto.index\_add\__ub

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

Serves as the UB version of [pypto.index_add_](pypto-index_add_.md). It has many constraints and does not guarantee stability.
It multiplies each data block of the source by the scaling factor alpha (default value: 1) and adds the result to the corresponding data block of the input. The index and data block direction are specified by index and dim. The following are examples:
$$
input[index[i], :, :] += alpha * source[i, :, :],\ if \ dim == 0, \\
input[:, index[i], :] += alpha * source[:, i, :],\ if \ dim == 1,\\
input[:, :, index[i]] += alpha * source[:, :, i],\ if \ dim == 2.
$$

## Prototype

```python
index_add__ub(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Destination operand (accumulated target).<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_FP32, DT_FP16, DT_BF16, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 5 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | An int specifying the dimension of input along which the addition is performed.<br>It supports any value not exceeding the number of dimensions of input. For details, see the "Constraints" section.|
| index   | Input     | Values representing the indices of input along the dim axis.<br>Supported type: Tensor.<br>The tensor supports the following data types: DT_INT32 and DT_INT64.<br>Empty tensors are not supported. Only 1-dimensional shapes are supported. The indices correspond one-to-one with the indices of source along the dim axis, and the shape size is the same as that of source along the dim axis.|
| source  | Input     | Source operand to be added to input.<br>Supported type: Tensor.<br>The data type of the tensor is the same as that of input.<br>Empty tensors are not supported. The shape supports 1 to 5 dimensions. The shape size along the dim axis is the same as that of index, and the shape sizes of other dimensions are the same as those of input. The shape size must not exceed 2147483647 (INT32_MAX).|
| alpha   | Input     | A scalar keyword argument.<br>It indicates the scaling factor for accumulation. The default value is 1.|

## Return Value

Returns input for in-place operations.

## Constraints

1. index must be of the integer type (DT\_INT32 or DT\_INT64), with values not exceeding the shape size of input along the dim axis. It must be 1-dimensional, and its shape size must be the same as that of the dim axis of source.

2. dim must be of the int type, with a value range of $-input.dim\leq dim < input.dim$.

3. input and source have the same data type and dimensions.

4. viewshape of input.shape and source.shape along the dim axis cannot be tiled, requiring $viewshape[dim]\geq\max (input.shape[dim], source.shape[dim])$. The shape sizes of the remaining dimensions are not restricted.

5. TileShape has the same dimensions as those of input. The dim axis and index of input and source cannot be tiled. The total size of all TileShape values for all inputs and outputs must not exceed the UB memory capacity.

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.
For example, if the input is [m, n, p], dim is 1, the source input is [m, t, p], the index input is [t], the output is [m, n, p], and TileShape is set to [m1, t1, p1], then m1 and p1 are used to tile the m and p axes, respectively. The n and t axes cannot be tiled and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
source = pypto.tensor([3, 3], pypto.DT_INT32)   # shape (3, 3)
index = pypto.tensor([3], pypto.DT_INT32)   # shape (3,)
dim = 0
# use alpha
y = pypto.index_add__ub(x, dim, index, source, alpha=1)
# not use alpha
y = pypto.index_add__ub(x, dim, index, source)
```

Result example:

```python
Input data x:   [[0 0 0],
               [0 0 0]]
      source: [[1 1 1],
               [1 1 1],
               [1 1 1]]
      index:   [0 1 0]
Output data y:   [[2 2 2],
               [1 1 1]]               # shape (2, 3)
```
