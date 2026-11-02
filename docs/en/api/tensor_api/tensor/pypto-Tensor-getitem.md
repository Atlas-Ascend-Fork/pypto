# pypto.Tensor Indexing

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:09:33.696Z pushedAt=2026-10-28T11:57:30.341Z nl=2 -->

Tensor indexing is one of the core operations for a tensor, used to filter, extract, or modify elements at specific positions in the tensor. Through indexing operations, you can precisely obtain partial data from a tensor (such as a single element, a sub-tensor, or data along a specific dimension), or assign values to modify elements at specified positions.

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

## I. \_\_getitem\_\_

## Description

Obtains a sub-tensor or a single element from a tensor by index or slice. This method supports multiple indexing modes, providing a flexible and intuitive way to access data.

## Prototype

```python
__getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None)
```

## Parameters

| Parameter | Input/Output | Description |
|-------------|-----------|----------------------------------------------------------------------|
| key | Input | Tensor index used to obtain the data at the corresponding position of a tensor.<br> Supported types:<br> - int or SymbolicScalar (symbolic scalar): a single integer index.<br> - slice: a slice object.<br> - tuple: a combination of multi-dimensional indices, whose types include int, SymbolicScalar, slice, and Ellipsis(...). |
| valid_shape | Input | Size of the valid data in the output tensor. |

## Return Value

Returns the tensor data at the corresponding index position.

## Constraints

1. For **slice** (a slice object in the format start:end:step), the **step** setting is not currently supported, and its value is fixed to **1** by default.

Unsupported syntax: a\[1:2:2, :\].

2. Boolean type indexing is not currently supported.

Unsupported syntax: a\[True, False, True, False\].

3. Tensor type indexing is not currently supported.

Unsupported syntax: a\[b\], \(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\).

## Example

1. Full slice

   Use slicing to obtain a subregion of the tensor.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[:2, :2] # Equivalent to view(a, [2, 2], [0, 0]).
   ```

   The results are as follows:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: [[1, 2],
               [5, 6]]
   ```

2. Mixed indexing and slicing

   Combining integer indexing and slicing reduces the dimensionality and extracts specific rows or columns.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[1, 1:3] # Equivalent to view(a, [1, 2], [1, 1]) first, and then reshape to [2].
   ```

   The results are as follows:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: [6, 7]
   ```

3. Negative indexing

    Supports Python-style negative indexing, counting from the end.

    ```python
    a = pypto.tensor([4, 4], pypto.DT_FP32)
    b = a[-1, -3:-1] # Equivalent to s[3, 1:3].
    ```

    The results are as follows:

    ```python
    Input activation a: [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]
    Output activation b: [14, 15]
    ```

4. Ellipsis (...)

   Use `...` to automatically fill all intermediate dimensions, simplifying multi-dimensional indexing.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[..., 1:3] # Equivalent to s[:, 1:3].
   ```

   The results are as follows:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: [[2, 3],
               [6, 7],
               [10, 11],
               [14, 15]]
   ```

5. Single-element access

   Integer indexing retrieves a single element of the tensor (only the `DT\_INT32` type is supported).

   ```python
   a = pypto.tensor([4, 4], pypto.DT_INT32)
   b = a[0, 0] # Returns a SymbolicScalar.
   ```

   The results are as follows:

   ```python
   Input activation a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output activation b: 1
   ```

6. Gather operation

   When the index is in the form of `\[int:Tensor\]`, a gather operation is performed, where the int type corresponds to dim and the tensor type corresponds to index. This slicing syntax is equivalent to `Tensor.gather\(dim, index\)`.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   index = pypto.tensor([1, 4], pypto.DT_INT32)
   b = a[0:index] # Calls gather(a, 0, index).
   ```

   The results are as follows:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Input data index: [[0, 1, 2, 3]]
   Output data b: [[1, 6, 11, 16]]
   ```

## II. \_\_setitem\_\_

## Description

Assigns values to the specified positions of a tensor by index or slice.

## Prototype

```python
__setitem__(self, key, value)
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| key     | Input      | Tensor index used to obtain the data at the corresponding position of a tensor.<br> Supported types:<br> - int or SymbolicScalar (symbolic scalar): a single integer index.<br> - slice: a slice object.<br> - tuple: a combination of multi-dimensional indices, whose types include int, SymbolicScalar, slice, and Ellipsis(...). |
| value   | Input      | Value to be set, whose type supports Tensor or a scalar (float/int). |

## Return Value

Returns the tensor after the values are assigned to the corresponding positions.

## Constraints

1. For **slice** (a slice object in the format start:end:step), the **step** setting is temporarily not supported by the function, and its value is fixed to **1** by default.

Unsupported syntax: a\[1:2:2, :\].

2. Boolean type indexing is currently not supported.

Unsupported syntax: a\[True, False, True, False\].

3. Tensor type indexing is not currently supported.

Unsupported syntax: a\[b\], \(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\).

## Example

1. Full slice

   Use slicing to assemble a small tensor into a specified position of a large tensor.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[0:, 0:] = b #Equivalent to assemble(b, (0, 0), a)
   ```

   The results are as follows:

   ```python
   Input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   Input data b: [[10, 10]
               [10, 10]]
   Output data a: [[10, 10, 0, 0],
               [10, 10, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

2. Mixed indexing and slicing

   Combine integer indexing and slicing to operate on specific rows or columns.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[0, 1:3] = b #b is reshaped to (1, 2), equivalent to pypto.assemble(b, (0, 1), a)
   ```

   The results are as follows:

   ```python
   Input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   Input data b: [10, 10]
   Output data a: [[0, 10, 10, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

3. Negative indexing

   Supports Python-style negative indexing, counting from the end.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[-1, -3:-1] = b #equivalent to a[3, 1:3]
   ```

   The results are as follows:

   ```python
   Input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   Input data b: [10, 10]
   Output data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 10, 10, 0]]
   ```

4. Ellipsis (...)

   Using ... automatically fills the intermediate dimensions.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[..., 2:4] = b #equivalent to a[0:2, 2:4]
   ```

   The results are as follows:

   ```python
   Input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   Input data b: [[10, 10]
               [10, 10]]
   Output data a: [[0, 0, 10, 10],
               [0, 0, 10, 10],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

5. Single-element assignment

   Integer indexing assigns a value to a single element (only DT\_INT32 is supported).

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_INT32)
   a[2, 3] = 5 # Call SetTensorData.
   ```

   The results are as follows:

   ```python
   Input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   Output data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 5],
               [0, 0, 0, 0]]
   ```

6. Scatter operation

   When **key** is a slice, **key.start** is an int, and **key.stop** is a tensor \(a\[start:stop\]   \), a scatter operation is performed.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   indices = pypto.Tensor([1, 4], pypto.DT_INT32) # Index tensor.
   values = pypto.Tensor([1, 4], pypto.DT_FP32)
   # Perform scatter on dimension 0.
   a[0:indices] = values # Call pypto.scatter(a, 0, indices, values).
   ```

   The results are as follows:

   ```python
   Input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   Input data indices: [[0, 1, 2, 3]]
   Input data values: [[10, 10, 10, 10]]
   Output data a: [[10, 0, 0, 0],
               [0, 10, 0, 0],
               [0, 0, 10, 0],
               [0, 0, 0, 10]]
   ```
