# pypto.index\_add\_

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:49:25.903Z pushedAt=2026-10-28T11:57:29.940Z nl=2 -->

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

Multiplies each data block of **source** by the scaling factor **alpha** (defaulting to 1) and adds it to the corresponding data block of **input**, where the index and data block direction are specified by **index** and **dim**. For example,
$ input[index[i], :, :] += alpha * source[i, :, :],\ if \ dim == 0, \\
    input[:, index[i], :] += alpha * source[:, i, :],\ if \ dim == 1,\\
    input[:, :, index[i]] += alpha * source[:, :, i],\ if \ dim == 2.$

## Prototype

```python
index_add_(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. Only 1- to 5-dimensional shapes are supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim     | Input      | An integer specifying the dimension of **input** along which the addition is performed.<br>Supports any value not exceeding the number of dimensions of **input**. For details, see the "Constraints" section. |
| index   | Input      | Source operand whose values represent the indices of **input** along the **dim** axis.<br>Supported type: Tensor.<br>Supported data types: DT_INT32 and DT_INT64.<br>Empty tensors are not supported. Only 1-dimensional shapes are supported. The indices correspond one-to-one with the indices of **source** along the **dim** axis, and the shape size is the same as that of **source** along the **dim** axis. |
| source  | Input      | Source operand to be added to **input**.<br>Supported type: Tensor.<br>The data type of the tensor is the same as that of **input**.<br>Supports 1- to 5-dimensional shapes. The shape size along the **dim** axis is the same as that of **index**, and the shape sizes of other dimensions are the same as those of **input**. |
| alpha   | Input      | A scalar keyword argument.<br>Scaling factor for accumulation, defaulting to 1. |

## Return Value

The in-place operation returns **input**.

## Constraints

1. `index` must be of integer type (DT_INT32 or DT_INT64), with values not exceeding the shape size of `input` along the `dim` axis. It must be 1-dimensional, and its shape size must match that of the `dim` axis of `source`.

2. **dim** is of the **int** type, with a value range of $-input.dim\leq dim < input.dim$.

3. **input** and **source** have the same data type and dimensionality.

4. The ViewShape of the non-**dim** axes of **input.shape** and **source.shape** cannot be tiled, that is, $ViewShape[i] \geq input.shape[i]=source.shape[i], i \ne dim$.

5. The dimensionality of the TileShape is the same as that of **source**, and it is used only to tile **source** and **index**. The total size of the TileShapes of all inputs and outputs must not exceed the size of the UB memory.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

For an input `[m, n, p]` with `dim=1`, a source of `[m, t, p]`, and an index of `[t]`, the output is `[m, n, p]`. If the `TileShape` is set to `[m1, t1, p1]`, then `m1`, `t1`, and `p1` are used to tile the `m`, `t`, and `p` axes of the source, respectively.

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
y = pypto.index_add_(x, dim, index, source, alpha=1)
# not use alpha
y = pypto.index_add_(x, dim, index, source)
```

The results are as follows:

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
