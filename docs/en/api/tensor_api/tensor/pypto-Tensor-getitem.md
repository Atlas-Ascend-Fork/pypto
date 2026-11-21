# pypto.Tensor.\_\_getitem\_\_

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
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

Obtains a sub-tensor or a single element from a tensor by indexing or slicing. This method supports multiple indexing modes and provides a flexible and intuitive way to access data.

## Prototype

```python
__getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None)
```

## Parameters

| Parameter     | Input/Output| Description                                                                |
|-------------|-----------|----------------------------------------------------------------------|
| key         | Input     | Tensor index, which is used to obtain the data at the corresponding position of the tensor.<br> The options are as follows:<br> - int or SymbolicScalar (symbolic scalar): a single integer index.<br> - slice: slice object.<br> - tuple: combination of multi-dimensional indexes, including int or SymbolicScalar, slice, and Ellipsis(...).|
| valid_shape | Input     | Size of the valid data in the output tensor.|

## Return Value

- Returns the single element at the corresponding position if all indexes are integers (int or SymbolicScalar). The element type is SymbolicScalar.
- Returns the sub-tensor at the corresponding index position if the index contains a slice.

## Constraints

1. For slice (slice object, in the format of start:end:step), the step parameter is not supported currently. The default value is fixed at 1.

   The following example is not supported: a\[1:2:2, :\].

2. Currently, indexes of the bool type are not supported.

   The following example is not supported: a\[True, False, True, False\].

3. Currently, indexes of the tensor type are not supported.

   Unsupported example: a\[b\], \(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)\).

4. The ellipsis (...) can appear at most once in an index, and the number of indexes cannot exceed the number of dimensions of the tensor.

## Examples

1. Empty slice

   Use an empty slice to obtain the entire tensor and return itself.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[:] # Returns a itself.
   ```

   Result example:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   ```

2. Full slicing (slice)

   Use a slice to obtain a sub-region of the tensor.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[:2, :2] # Equivalent to view(a, [2, 2], [0, 0])
   ```

   Result example:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: [[1, 2],
               [5, 6]]
   ```

3. Default index

   If the number of indexes is less than the number of dimensions of the tensor, the missing dimensions are automatically padded with full slices (:).

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[1] # Equivalent to a[1, :]
   ```

   Result example:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: [5, 6, 7, 8]
   ```

4. Mixed index and slice

   You can combine integer indexes and slices to reduce the number of dimensions and extract specific rows or columns.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[1, 1:3] # Equivalent to view(a, [1, 2], [1, 1]) and then reshaped to [2].
   ```

   Result example:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: [6, 7]
   ```

5. Negative indexing

    Python-style negative indexing is supported, counting from the end.

    ```python
    a = pypto.tensor([4, 4], pypto.DT_FP32)
    b = a[-1, -3:-1] # Equivalent to a[3, 1:3]
    ```

    Result example:

    ```python
    Input data a: [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]
    Output data b: [14, 15]
    ```

6. Ellipsis (...)

   Uses \`...\` to automatically fill in all intermediate dimensions, simplifying multidimensional indexing.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[..., 1:3] # Equivalent to a[:, 1:3]
   ```

   Result example:

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

7. Single-element access

   Integer index, used to obtain a single element from a tensor (only the DT\_INT32 type is supported).

   ```python
   a = pypto.tensor([4, 4], pypto.DT_INT32)
   b = a[0, 0] # Return SymbolicScalar.
   ```

   Result example:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Output data b: 1
   ```

8. Gather operation

   When the index is in the form of \[int:Tensor\], the gather operation is performed. The int type in the index corresponds to dim, and the Tensor type corresponds to index. This slicing syntax is equivalent to Tensor.gather\(dim, index\).

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   index = pypto.tensor([1, 4], pypto.DT_INT32)
   b = a[0:index] # Call gather(a, 0, index).
   ```

   Result example:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Input data index: [[0, 1, 2, 3]]
   Output data b: [[1, 6, 11, 16]]]
   ```
