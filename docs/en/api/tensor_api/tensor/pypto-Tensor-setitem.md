# pypto.Tensor.\_\_setitem\_\_

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

Assigns a value to a specified position in a tensor by using an index or slice.

## Prototype

```python
__setitem__(self, key, value)
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| key     | Input     | Tensor index, which is used to locate the position to which a value is to be assigned in the tensor.<br> The options are as follows:<br> - int or SymbolicScalar (symbolic scalar): a single integer index.<br> - slice: slice object<br> - tuple: combination of multi-dimensional indexes, including int or SymbolicScalar, slice, and Ellipsis(...).|
| value   | Input     | Value to be set.<br> The options are as follows:<br> - Tensor: used for slice or mixed index assignment. The shape must match the region covered by the index.<br> - Scalar (int or SymbolicScalar): used for single-element assignment. The target tensor must be of the DT\_INT32 type.|

## Return Value

No value is returned. The target tensor is modified in place.

## Constraints

1. For slice (slice object, in the format of start:end:step), the step parameter is not supported currently. The default value is fixed at 1.

   The following example is not supported: a\[1:2:2, :\].

2. Currently, indexes of the bool type are not supported.

   The following example is not supported: a\[True, False, True, False\].

3. Currently, indexes of the tensor type are not supported.

   Unsupported example: a\[b\], \(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)\).

4. When a hybrid index (a combination of int/SymbolicScalar and slice) is used to assign values, the number of slices in the key must be the same as the number of dimensions of value.

5. When a single-element value is assigned, value can only be of the int or SymbolicScalar type. Currently, the float type is not supported.

## Examples

1. Empty slice

   Using an empty slice to assign values to the entire tensor is equivalent to moving value to the target tensor. This operation is often used to assign the computation result to the output tensor. If value and the target tensor have the same shape, the final data effect is equivalent to full assembly.

   ```python
   @pypto.frontend.jit()
   def add_kernel(
      a: pypto.Tensor[[], pypto.DT_INT32], b: pypto.Tensor[[], pypto.DT_INT32], c: pypto.Tensor[[], pypto.DT_INT32]
   ):
      pypto.set_vec_tile_shapes(32, 32)
      c[:] = a + b
   ```

   Result example:

   ```python
   Input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   Input data b: [[1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1]]
   Output data c: [[2, 3, 4, 5],
               [6, 7, 8, 9],
               [10, 11, 12, 13],
               [14, 15, 16, 17]]
   ```

2. Full slice

   Use a slice to assemble a small tensor to the specified position in a large tensor.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[0:, 0:] = b # Equivalent to assemble(b, (0, 0), a)
   ```

   Result example:

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

3. Hybrid index and slice

   You can use integer indexing and slicing together to perform operations on specific rows or columns.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[0, 1:3] = b # b is reshaped to (1, 2), which is equivalent to pypto.assemble(b, (0, 1), a).
   ```

   Result example:

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

4. Negative index

   Python-style negative indexing is supported, counting from the end.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[-1, -3:-1] = b # Equivalent to a[3, 1:3]
   ```

   Result example:

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

5. Ellipsis (...)

   Using... can automatically fill in the middle dimensions.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[..., 2:4] = b # Equivalent to a[0:2, 2:4]
   ```

   Result example:

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

6. Single-element value assignment

   Integer index, used to assign a value to a single element (only the DT\_INT32 type is supported).

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_INT32)
   a[2, 3] = 5 # Call SetTensorData.
   ```

   Result example:

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

7. Scatter operation

   When key is slice, key.start is int, and key.stop is tensor \(a\[start:stop\]   \), perform the scatter operation.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   indices = pypto.Tensor([1, 4], pypto.DT_INT32) # Index tensor
   values = pypto.Tensor([1, 4], pypto.DT_FP32)
   # Perform scatter on dimension 0.
   a[0:indices] = values # Call pypto.scatter(a, 0, indices, values).
   ```

   Result example:

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
