# pypto.index\_put\_

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:52:12.993Z pushedAt=2026-10-28T11:57:29.953Z nl=2 -->

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

Updates the tensor `self` with one or more blocks of data from `values` at positions specified by `indices`. If `accumulate` is set to `True`, the update operation performs an element-wise addition between `values` and the existing entries at the corresponding positions; otherwise, if `accumulate` is `False`, the existing entries are directly overwritten.

## Prototype

```python
index_put_(input: Tensor, indices: tuple, values: Tensor, accumulate: bool = False) -> None
```

## Parameters

|   Parameter   | Input/Output | Description                                                                  |
|------------|-----------|----------------------------------------------------------------------|
|   input    |    Input   | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FP16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |
|  indices   |   Input    | A tuple of **Tensor** type, where each tensor represents the index of one dimension.<br>Supported type: tuple\[Tensor\], where each Tensor is one-dimensional and has the same dimension.<br>Supported data types of Tensor: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, and DT_UINT64.<br>Empty tensors are not supported. The number of tensors in the tuple does not exceed the number of dimensions of **input**. |
|   values   |   Input    | Values to be updated into **input**.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FP16, and DT_FP32.<br>Empty tensors are not supported. The number of dimensions does not exceed that of **input**. |
| accumulate |   Input (optional)    | Accumulation parameter, defaulting to **False**.<br>Supported type: bool. |

## Return Value

Performs an in-place operation on **input** and returns no value.

## Constraints

1. All 1D tensors in `indices` must have the same shape, and broadcasting is not supported. The values in the i-th tensor of `indices` must be less than the shape size of the `input` tensor along its (i-1)-th dimension. If multiple selections in `indices` refer to the same position for updates, the result is undefined.

2. **values** does not support broadcast, and the shape of its 0th dimension must be the same as the shape of the one-dimensional tensor in **indices**. If the dimension of **values** is greater than or equal to 2, the last i dimensions (i>0) excluding the 0th dimension are exactly the same as the shape of the last i dimensions of **input**.

3. The dimension of **input**, the number of tensors in **indices**, and the dimension of **values** must satisfy: (input.shape.size) + 1 = (indices.size) + (values.shape.size).

4. The data types of **input** and **values** must be the same.

5. **viewshape** is one-dimensional and is used for tiling each one-dimensional tensor in **indices** and the 0-th dimension of **values**; the other dimensions of **values** are not tiled.

6. The dimensionality of `TileShape` must not exceed that of `values`, and tiling is applied to `values` along each 1-dimensional tensor in `indices`. The total size of `TileShape` for `indices` and `values` combined must not exceed the UB memory capacity.

7. When **accumulate** is **True**, the data types of **input** and **values** support only DT_BF16, DT_FP16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32.

8. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensionality of `TileShape` must not exceed that of `values`. If `TileShape` has fewer dimensions than `values`, the missing trailing dimensions are automatically padded to match the corresponding dimensions of `values` during tiling.

For example, given `input` of `[m, n, p]`, `indices` as `([t])`, and `values` of `[t, n, p]`, if `TileShape` is set to `[t1, n1, p1]`, then `t1` tiles the `t` axis, `n1` tiles the `n` axis, `p1` tiles the `p` axis, while the `m` axis is not tiled.

If `input` is `[m, n, p]`, `indices` is `([t])`, and `values` is `[t, n, p]`, with `TileShape` set to `[t1, n1]`, then `TileShape` is automatically padded to `[t1, n1, p]`, where `t1` tiles the `t` axis and `n1` tiles the `n` axis, while the `m` and `p` axes are not tiled.

If `input` is `[m, n, p]`, `indices` is `([t], [t])`, and `values` is `[t, p]`, with `TileShape` set to `[t1, p1]`, then `t1` tiles the `t` axis and `p1` tiles the `p` axis, while the `m` and `n` axes are not tiled.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([3, 3], pypto.DT_INT32)
indices0 = pypto.tensor([2], pypto.DT_INT32)
indices = (indices0, )
values = pypto.tensor([2, 3], pypto.DT_INT32)
accumulate = True
# accumulate is True
pypto.index_put_(x, indices, values, accumulate)
# accumulate is False(default)
pypto.index_put_(x, indices, values)
```

The results are as follows:

```python
Input data x:      [[1 1 1],
                 [1 1 1],
                 [0 0 0]]
      indices:   ([1 2], )
      values:    [[0 1 0],
                  [0 2 0]]
x after in-place update:   [[1 1 1],
                 [1 2 1],
                 [0 2 0]]               # accumulate is True
                 [[1 1 1],
                 [0 1 0],
                 [0 2 0]]               # accumulate is False
```
