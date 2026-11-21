# pypto.index\_put\_

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
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id11 -->
- Kirin 9050 processors: Supported
<!-- end id11 -->

## Description

Updates self with one or more blocks of data from values at positions specified by indices. If accumulate is set to True, the update operation performs an element-wise addition between values and the existing entries at the corresponding positions; otherwise, if accumulate is False, the existing entries are directly overwritten.

## Prototype

```python
index_put_(input: Tensor, indices: tuple, values: Tensor, accumulate: bool = False) -> None
```

## Parameters

|   Parameter  | Input/Output| Description                                                                 |
|------------|-----------|----------------------------------------------------------------------|
|   input    |    Input  | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FP16, and DT_FP32. <!-- npu="x90,9030,9050" id8 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id8 --><br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
|  indices   |   Input   | A tuple of the Tensor type, where each tensor represents the index of one dimension.<br>Supported type: tuple\[Tensor\]. Each tensor is one-dimensional and has the same dimension.<br>A tensor supports the following data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, and DT_UINT64. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id9 --><br>Empty tensors are not supported. The number of tensors in the tuple cannot be greater than the number of dimensions of the input.|
|   values   |   Input   | Values to be updated into input.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FP16, and DT_FP32. <!-- npu="x90,9030,9050" id10 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id10 --><br>Empty tensors are not supported. The number of dimensions cannot exceed that of input.|
| accumulate |   Input (optional)   | Accumulation parameter, defaulting to False.<br>Supported type: bool.|

## Return Value

Performs in-place operations on input and returns no value.

## Constraints

1. All one-dimensional tensors in indices must have the same shape and broadcasting is not supported. The values in the ith tensor of indices must be smaller than the shape size of the input tensor along its (i-1)th dimension. If multiple selections in indices refer to the same position for updates, the result is undefined.

2. values does not support broadcasting. The shape of its 0th dimension must be the same as that of the one-dimensional tensor in indices. If the number of dimensions of values is greater than or equal to 2, the last i dimensions (i>0) of values (excluding the 0th dimension) must have the same shape as the last i dimensions of input.

3. The dimension of input, the number of tensors in indices, and the dimension of values must satisfy (input.shape.size) + 1 = (indices.size) + (values.shape.size).

4. The data types of input and values must be the same.

5. viewshape is one-dimensional. It is used for tiling each one-dimensional tensor in indices and the 0th dimension of values; the other dimensions of values are not tiled.

6. The dimensions of TileShape must not exceed those of values. Tiling is applied to values along each 1-dimensional tensor in indices. The total size of TileShape for indices and values combined must not exceed the size of the UB memory.

7. When accumulate is True, the data types of input and values support DT_BF16, DT_FP16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32.

8. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

<!-- npu="x90,9030,9050" id6 -->
9. A tensor supports the following data types:
   <!-- npu="9030" id12 -->
   - Kirin 9030 processors:
      - input/values: DT_FP16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32
      - indices: DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64
   <!-- end id12 -->
   <!-- npu="x90" id13 -->
   - Kirin X90 processors:
      - input/values: DT_FP16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32
      - indices: DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64
   <!-- end id13 -->
   <!-- npu="9050" id16 -->
   - Kirin 9050 processors:
      - input/values: DT_FP16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32
      - indices: DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, and DT_UINT64
   <!-- end id16 -->
<!-- end id6 -->

<!-- npu="x90,9030,9050" id7 -->
10. accumulate support:
    <!-- npu="9030" id14 -->
    - Kirin 9030 processors: Not supported. This parameter must be set to False.
    <!-- end id14 -->
    <!-- npu="x90" id15 -->
    - Kirin X90 processors: Not supported. This parameter must be set to False.
    <!-- end id15 -->
    <!-- npu="9050" id17 -->
    - Kirin 9050 processors: Not supported. This parameter must be set to False.
    <!-- end id17 -->
<!-- end id7 -->

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

The number of dimensions of TileShape must not exceed that of values. If TileShape has fewer dimensions than values, the missing dimensions are automatically padded during tiling to match the shape of values.

If the input is [m, n, p], indices is [t], values is [t, n, p], and TileShape is set to [t1, n1, p1], t1 is used to tile the t axis, n1 is used to tile the n axis, p1 is used to tile the p axis, and the m axis is not tiled.

If the input is [m, n, p], indices is [t], values is [t, n, p], and TileShape is set to [t1, n1], TileShape is automatically padded to [t1, n1, p]. t1 is used to tile the t axis, n1 is used to tile the n axis, and the m and p axes are not tiled.

If the input is [m, n, p], indices is ([t], [t]), values is [t, p], and TileShape is set to [t1, p1], t1 is used to tile the t axis, p1 is used to tile the p axis, and the m and n axes are not tiled.

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

Result example:

```python
Input data x:      [[1 1 1],
                 [1 1 1],
                 [0 0 0]]
      indices:   ([1 2], )
      values:    [[0 1 0],
                  [0 2 0]]
Updated x:   [[1 1 1],
                 [1 2 1],
                 [0 2 0]]               # accumulate is True
                 [[1 1 1],
                 [0 1 0],
                 [0 2 0]]               # accumulate is False
```
