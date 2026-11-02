# Tensor Operations

<!-- md-trans-meta sourceCommit=50483671e36b55446f83f1726cc2e05a8b7b04bd translatedAt=2026-10-27T02:35:41.070Z pushedAt=2026-10-27T09:02:59.895Z nl=2 -->

## Mathematical Operations

PyPTO provides a comprehensive set of operations for tensor computation, designed to deliver efficient and flexible computing capabilities. These operations are categorized into vector operations and cube operations, and can be combined to implement more complex computation logic.

### Vector Operations

Vector operations perform element-wise computations on tensors, applicable to various basic mathematical operations.

- Arithmetic operations

    ```python
    # Addition
    result = pypto.add(a, b)
    result = pypto.add(a, scalar)  # Add scalar to tensor
    result = a + b

    # Subtraction
    result = pypto.sub(a, b)
    result = pypto.sub(a, scalar)
    result = a - b

    # Multiplication
    result = pypto.mul(a, b)
    result = pypto.mul(a, scalar)
    result = a * b

    # Division
    result = pypto.div(a, b)
    result = pypto.div(a, scalar)
    result = a / b

    # Exponentiation
    result = pypto.pow(a, scalar)  # a ** scalar
    ```

- Mathematical functions

    ```python
    # Exponential and logarithm
    result = pypto.exp(x)      # e ** x
    result = pypto.log(x)      # ln(x)
    result = x.exp()
    result = x.log()

    # Square root
    result = pypto.sqrt(x)     # √x
    result = pypto.rsqrt(x)    # 1/√x
    result = x.sqrt()
    result = x.rsqrt()

    # Trigonometric functions
    result = pypto.sin(x)
    result = pypto.cos(x)
    result = x.sin()
    result = x.cos()

    # Absolute value
    result = pypto.abs(x)
    result = x.abs()

    # Opposite number
    result = pypto.neg(x)      # -x
    result = x.neg()
    ```

- Activation functions

    ```python
    # Sigmoid
    result = pypto.sigmoid(x)  # 1 / (1 + exp(-x))
    result = x.sigmoid()

    # ReLU
    result = pypto.relu(x)     # Maximum of (0, x)
    result = x.relu()

    # Softmax
    result = pypto.softmax(x, dim=-1)  # Apply Softmax along the dim dimension
    result = x.softmax(dim=-1)
    ```

- Comparison operations

    ```python
    # Maximum and minimum
    result = pypto.maximum(a, b)  # Element-wise maximum
    result = pypto.minimum(a, b)  # Element-wise minimum

    # Clipping
    result = pypto.clip(x, min_val, max_val)  # Clip the value range of x
    ```

- Reduction operations

    ```python
    # Summation
    result = pypto.sum(x, dim=-1, keepdim=False)
    result = x.sum(dim=-1, keepdim=False)

    # Maximum value
    result = pypto.amax(x, dim=-1, keepdim=False)
    result = x.amax(dim=-1, keepdim=False)

    # Minimum value
    result = pypto.amin(x, dim=-1, keepdim=False)
    result = x.amin(dim=-1, keepdim=False)
    ```

- In-place modification \(inplace\)

    ```python
    # Use move() for in-place operation
    output.move(pypto.add(a, b))  # Efficient, no copy

    # Or use assignment
    output[:] = pypto.add(a, b)   # Also efficient
    ```

- Broadcast

    Many operations support broadcast, making operations more flexible and efficient.

    ```python
    # Tensor + scalar (tensor as scalar)
    result = pypto.add(tensor, 2.0)

    # Tensor + 1D tensor (1D tensor)
    bias = pypto.tensor([features], pypto.DT_BF16, "bias")
    result = pypto.add(tensor, bias)
    ```

### Cube Operations

Cube operations are optimized for the Cube core of the NPU and are suitable for large-scale cube computation.

```python
# Basic cube multiplication
# C = A @ B, where A: [M, K], B: [K, N], C: [M, N]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)

# Use transpose, where A: [M, K], B: [N, K]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, a_trans=False, b_trans=True)

# Batch cube multiplication
# A: [B, M, K], B: [B, K, N], result: [B, M, N]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)

# With bias
bias = pypto.tensor([1, N], pypto.DT_BF16, "bias")
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, extend_params={'bias_tensor': bias})

# Use transpose, where A: [M, K], B: [N, K], output in NZ format
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, a_trans=False, b_trans=True, c_matrix_nz=True)
```

Cube multiplication parameters:

- `input`: Left cube \[M, K\] or \[B, M, K\]
- `mat2`: Right cube \[K, N\] or \[B, K, N\]
- `out_dtype`: Output data type
- `a_trans`: Transpose left cube (default: False)
- `b_trans`: Transpose right cube (default: False)
- `c_matrix_nz`: Output in NZ format (default: False)
- `extend_params`: Extended features (bias, dequantization, etc.).

### Combined Operations

You can combine the preceding basic operations to implement more complex computation logic.

```python
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    row_max = pypto.amax(x, dim=-1, keepdim=True)  # Compute the row maximum
    sub = x - row_max                              # Normalize the values
    exp = pypto.exp(sub)                           # Exponential operation
    esum = pypto.sum(exp, dim=-1, keepdim=True)    # Sum
    return exp / esum                              # Normalize the probabilities

def softmax_kernel(x: pypto.Tensor, y: pypto.Tensor) -> None:
    ...
    for idx in pypto.loop(x.shape[0]):
        ...
        softmax_out = softmax_core(x)
        ...
```

## Logical Structure Transformations

Logical structure transformation operations allow users to perform transformations on tensors, such as shape, dimension, and type changes, to meet different computational requirements.

### View and Assemble

- View: A view operation creates a new tensor reference to the same underlying data, suitable for block processing and local computation.

    ```python
    # View
    view = pypto.view(tensor, view_shape, offset, valid_shape)
    ```

    Example:

    ```python
    # Create a view with a specific shape and offset
    view = pypto.view(
        tensor,
        view_shape=[32, 32],
        offset=[10, 20],
        valid_shape=[actual_h, actual_w]  # Optional
    )
    ```

    Parameters:

    - tensor: source tensor
    - view\_shape: shape of the view
    - offset: start position of the source tensor
    - valid\_shape: actual valid size (for boundary handling)

    Tensor supports Python-style indexing and slicing for flexible data access.

    ```python
    tensor = pypto.tensor([10, 20], pypto.DT_FP16, "tensor")

    # Single element (create a view)
    element = tensor[0, 0]  # Only INT32 is supported

    # Slice (create a view)
    slice_tensor = tensor[0:5, 10:20]

    # Ellipsis
    ellipsis_slice = tensor[..., 0:10]
    ```

- Assemble: The assemble function places a smaller tensor into a larger tensor at a specified offset, suitable for merging results after block processing.

    ```python
    # Assemble a small tensor into a large tensor
    pypto.assemble(
        small_tensor,      # Source tensor
        [10, 20],          # Target position
        large_tensor       # Target tensor
    )
    ```

    Example:

    ```python
    # Small tensor result
    tile_result = pypto.tensor([32, 32], pypto.DT_FP16, "tile")

    # Large output tensor
    output = pypto.tensor([100, 200], pypto.DT_FP16, "output")

    # Assemble the output tensor at position [10, 20]
    pypto.assemble(tile_result, [10, 20], output)
    ```

### Reshaping Shape and Dimensions

```python
# Reshape the tensor
reshaped = pypto.reshape(tensor, [new_shape])

# Transpose
transposed = pypto.transpose(tensor, dim0=0, dim1=1)
```

Reshaping does not change the data; it only changes the view of the tensor.

### Type Casting

```python
# Convert to a different data type
result = pypto.cast(tensor, pypto.DT_FP32, mode=pypto.CastMode.CAST_NONE)
```
