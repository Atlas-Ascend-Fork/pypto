# FC0XXX-FC2XXX

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:33:36.869Z pushedAt=2026-10-27T04:07:08.430Z nl=2 -->

## FC0000 ERR_PARAM_INVALID

**Error Description**

Invalid vector input parameter error, for example, the parameter value, dimension, or format does not satisfy the constraints.

**Possible Causes**

- Shape constraint not satisfied: The input/output tensor is an empty tensor (a dimension is 0), or the shape size is greater than 2147483647 (INT32_MAX).

   ```python
   # Error example - Shape size greater than INT32_MAX
   a = pypto.tensor([65536, 65536], pypto.DT_FP16)  # 65536*65536>INT32_MAX
   b = pypto.tensor([65536, 65536], pypto.DT_FP16)
   out = pypto.pow(a, b)  # Triggers ERR_PARAM_INVALID
   ```

- Reduction axis constraints not satisfied: The dim of reduction operators (such as amax) exceeds the input dimension range.

   ```python
   # Error example - amax dim exceeds the input dimension range
   x = pypto.tensor([2, 3], pypto.DT_FP32)  # 2 dimensions
   out = pypto.amax(x, 2)  # dim=2 is out of range; only 0, 1, or negative indices are supported
   ```

**Solution**

1. Check the corresponding operator documentation (such as [pypto.add](../../../api/tensor_api/operation/pypto-add.md), [pypto.sin](../../../api/tensor_api/operation/pypto-sin.md), [pypto.cast](../../../api/tensor_api/operation/pypto-cast.md), and [pypto.amax](../../../api/tensor_api/operation/pypto-amax.md)) to ensure that the input/output shapes, dimensions, and other parameters meet the requirements.

   ```python
   # Correct example - input is a 2-dimensional Tensor
   a = pypto.tensor([4, 4], pypto.DT_FP16)
   b = pypto.tensor([4, 4], pypto.DT_FP16)
   out = pypto.add(a, b)
   ```

2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC0001 ERR_PARAM_DTYPE_UNSUPPORTED

**Error Description**

The vector input parameter data type is unsupported: a dtype that is not supported by the current operator or hardware is used.

**Possible Causes**

- The operator data type constraints are not satisfied: the input dtype is not in the supported set of the operator. For details about the data types supported by each operator, see the corresponding operator documentation.

   ```python
   # Error example: sin supports only DT_FP32/DT_FP16, but DT_INT64, which is not supported by sin, is passed in
   x = pypto.tensor([4], pypto.DT_INT64)
   out = pypto.sin(x)  # Triggers ERR_PARAM_DTYPE_UNSUPPORTED
   ```

- Input data type consistency constraint not satisfied: The two input tensors of a binary operation have inconsistent data types.

   ```python
   # Error example: the two input data types are inconsistent (DT_FP16+DT_FP32)
   a = pypto.tensor([4, 4], pypto.DT_FP16)
   b = pypto.tensor([4, 4], pypto.DT_FP32)
   out = pypto.add(a, b)
   ```

**Solution**

1. Check the corresponding operator documentation (such as [pypto.add](../../../api/tensor_api/operation/pypto-add.md), [pypto.sin](../../../api/tensor_api/operation/pypto-sin.md), and [pypto.cast](../../../api/tensor_api/operation/pypto-cast.md)) to confirm that the input/output data types meet the requirements, and switch to a compatible data type and try again.

   ```python
   # Correct example: sin uses the supported DT_FP32
   x = pypto.tensor([4], pypto.DT_FP32)
   out = pypto.sin(x)
   ```

2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC0002 ERR_PARAM_SHAPE_DIM_UNSUPPORTED

**Error Description**

The vector input shape dimension is unsupported: the number of dimensions of the input/output tensor is outside the range supported by the operator, or the number of dimensions is inconsistent among multiple inputs.

**Possible Causes**

- Dimension constraints not satisfied: The number of dimensions of the input/output tensor is outside the range supported by the operator (for example, most vector operators support only 1 to 4 dimensions).

   ```python
   # Error example: the input has 5 dimensions, exceeding the 1-4 dimension range supported by Vector
   a = pypto.tensor([2, 2, 2, 2, 2], pypto.DT_FP16)  # 5 dimensions
   b = pypto.tensor([2, 2, 2, 2, 2], pypto.DT_FP16)
   out = pypto.pow(a, b)  # Triggers ERR_PARAM_SHAPE_DIM_UNSUPPORTED
   ```

- Dimension mismatch constraint not satisfied: The number of dimensions of the tensors involved in the operation is inconsistent.

   ```python
   # Error example: the two inputs have inconsistent numbers of dimensions (2D and 3D)
   a = pypto.tensor([4, 4], pypto.DT_FP16)        # 2 dimensions
   b = pypto.tensor([2, 4, 4], pypto.DT_FP16)     # 3 dimensions
   out = pypto.pow(a, b)
   ```

**Solution**

1. Check the corresponding operator documentation (such as [pypto.pow](../../../api/tensor_api/operation/pypto-pow.md), [pypto.gcd](../../../api/tensor_api/operation/pypto-gcd.md), and [pypto.amax](../../../api/tensor_api/operation/pypto-amax.md)) to confirm the supported dimension range, and ensure that all input tensors have the same number of dimensions within the supported range.

   ```python
   # Correct example: the input is a 2D tensor and the numbers of dimensions are consistent
   a = pypto.tensor([4, 4], pypto.DT_FP16)
   b = pypto.tensor([4, 4], pypto.DT_FP16)
   out = pypto.pow(a, b)
   ```

2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC0003 ERR_PARAM_COUNT_INVALID

**Error Description**

Invalid number of vector input operands: During operator execution or output shape inference, the number of input/output operands detected does not match the expected count.

**Possible Causes**

- The operand count constraint is not satisfied: The number of input/output operands passed to the operator does not match the expected count. The error log typically contains a message like `iOperands.size() should be ...`. This often occurs when the calling method does not match the number of inputs/outputs required by the operator documentation, or when there is an internal operator wiring error in the framework.

**Solution**

1. Compare the expected number of operands in the screen log against the corresponding operator documentation (such as [pypto.scatter_](../../../api/tensor_api/operation/pypto-scatter_.md), [pypto.concat](../../../api/tensor_api/operation/pypto-concat.md), and [pypto.where](../../../api/tensor_api/operation/pypto-where.md)) to verify that the number of inputs/outputs matches the calling method.
2. If the error persists despite correct calling (suspected internal framework wiring issue), please file an [issue](https://gitcode.com/cann/pypto/issues) in the community.

## FC1000 ERR_CONFIG_TILE

**Error Description**

Invalid vector tile configuration.

**Possible Causes**

- Tile value constraints are not satisfied: a dimension passed to `set_vec_tile_shapes` is less than or equal to 0.

   ```python
   # Error example - the second dimension of TileShape is 0
   pypto.set_vec_tile_shapes(4, 0)  # Each dimension must be greater than 0
   a = pypto.tensor([4, 16], pypto.DT_FP16)
   out = pypto.sin(a)
   ```

- TileShape dimension count constraint not satisfied: The number of dimensions passed to `set_vec_tile_shapes` exceeds 4.

   ```python
   # Error example - TileShape has more than 4 dimensions
   pypto.set_vec_tile_shapes(1, 1, 1, 1, 1)  # At most 4 dimensions are allowed
   ```

- The TileShape size constraint is not satisfied: `set_vec_tile_shapes` sets a value that exceeds the TileShape size constraint of the operator.

   ```python
   # Error example - the TileShape of amax exceeds 64KB
    pypto.set_vec_tile_shapes(1024, 1024)  # Under FP16, 1024*1024*2Byte=2MB, exceeding 64KB
   x = pypto.tensor([1024, 1024], pypto.DT_FP16)
   out = pypto.amax(x, -1, True)
   ```

- The TileShape dimension consistency constraint is not satisfied: The number of TileShape dimensions does not match the number of output (or input) tensor dimensions.

   ```python
   # Error example - the TileShape dimensions do not match the input dimensions
   pypto.set_vec_tile_shapes(4, 16)  # 2-dimensional TileShape
   a = pypto.tensor([2, 2, 2, 2], pypto.DT_FP16)  # 4-dimensional input
   out = pypto.add(a, a)
   ```

**Solution**

1. Check [pypto.set_vec_tile_shapes](../../../api/tensor_api/config/pypto-set_vec_tile_shapes.md) and the corresponding operator documentation to confirm that the TileShape values meet the requirements.

2. Before calling [pypto.set_vec_tile_shapes](../../../api/tensor_api/config/pypto-set_vec_tile_shapes.md), ensure that each dimension is a positive number, the number of dimensions does not exceed 4, and it matches the output dimensions.

   ```python
   # Correct example - each TileShape dimension is positive and matches the input dimensions
   pypto.set_vec_tile_shapes(4, 16)
   a = pypto.tensor([4, 16], pypto.DT_FP16)
   out = pypto.sin(a)
   ```

3. Read back the effective TileShape through [pypto.get_vec_tile_shapes](../../../api/tensor_api/config/pypto-get_vec_tile_shapes.md) and check whether it meets the tile constraints:

   ```python
   pypto.set_vec_tile_shapes(4, 16)
   tile_shape_info = pypto.get_vec_tile_shapes()
   print(tile_shape_info)
   # Output: [4, 16]
   ```

4. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC1001 ERR_CONFIG_ALIGNMENT

**Error Description**

Vector alignment constraints not satisfied: The address or shape is not aligned as required by the hardware.

**Possible Causes**

- Last axis alignment constraint not satisfied: The last axis of the operator (such as amax) does not meet the 32-byte alignment constraint.

   ```python
   # Incorrect example - amax last axis not 32-byte aligned
    pypto.set_vec_tile_shapes(4, 10)  # Under FP16, 10*2Byte=20Byte, not 32-byte aligned
   x = pypto.tensor([4, 10], pypto.DT_FP16)
   out = pypto.amax(x, -1, True)
   ```

**Solution**

1. Check the corresponding operator documentation (such as [pypto.amax](../../../api/tensor_api/operation/pypto-amax.md)) and [pypto.set_vec_tile_shapes](../../../api/tensor_api/config/pypto-set_vec_tile_shapes.md) to confirm the alignment requirements. Pay attention to whether reshape/view, dimension swapping (transpose), etc. change the inner axis alignment requirements, and adjust the tensor shape or TileShape value if necessary.

   ```python
   # Correct example - amax last axis 32-byte aligned (16 elements under FP16)
   pypto.set_vec_tile_shapes(4, 16)
   x = pypto.tensor([4, 16], pypto.DT_FP16)
   out = pypto.amax(x, -1, True)
   ```

2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC2000 ERR_RUNTIME_NULLPTR

**Error Description**

Vector runtime null pointer: The input tensor is null.

**Possible Causes**

NA

**Solution**

1. Ensure that all input/output tensors passed to the vector API are non-null and have completed address allocation. Check whether any nullptr exists.

   ```python
   # Correct example - all input Tensors have data allocated
   a = pypto.tensor([16, 32], pypto.DT_FP16, "a")
   b = pypto.tensor([16, 32], pypto.DT_FP16, "b")
   out = pypto.add(a, b)
   ```

2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC2001 ERR_RUNTIME_LOGIC

**Error Description**

Vector runtime logic error: The computation flow enters an undefined or abnormal branch.

**Possible Causes**

NA

**Solution**

1. Locate the abnormal path through relevant logs, check whether the computation flow enters an undefined or abnormal branch, and verify whether intermediate results and index values meet expectations.

2. Check whether the preconditions of the core computation logic (such as TileShape settings, and context and configuration handle initialization) are satisfied.

3. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).
