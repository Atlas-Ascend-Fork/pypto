# pypto.axpy_

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

Performs the AXPY operation: `y = alpha * x + y`. This operation updates the y tensor in place.

The computation formula is as follows:

$$
y_i = \alpha \cdot x_i + y_i
$$

**Important note**: AXPY is an in-place operation, and the y tensor will be directly modified. If the original y value before the AXPY operation is required for subsequent computation, back it up with `pypto.clone(y)` before calling AXPY.

## Prototype

```python
axpy_(y: Tensor, x: Tensor, alpha: Union[int, float] = 1.0) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                                                                                                                                                                          |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| y      | Input/Output| Target tensor, which is updated in place.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32 and DT_FP16.<br>**Broadcasting not supported**: The shape of y must be able to accommodate the broadcast result of x. This means the dimension of y cannot be 1 (unless the corresponding dimension of x is also 1).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| x      | Input     | Source tensor, which can be broadcast to the shape of y.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32 and DT_FP16.<br>Broadcasting supported: x can be broadcast to the shape of y (for example, the shape of x is `[m, 1]` and the shape of y is `[m, n]`).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).                                                |
| alpha  | Input     | Scaling factor used to scale x.<br>The supported types are int and float. The default value is 1.0.<br>The data type of alpha is automatically converted to match that of y.                                                                                                           |

## Return Value

Returns the updated tensor y (which shares the same memory address as the input y), with the same data type and shape as y.

## Constraints

1. **dtype constraints**:
   - Same dtype: DT_FP32 + DT_FP32 and DT_FP16 + DT_FP16 are supported.
   - Mixed dtype: Only DT_FP32 (y) + DT_FP16 (x) is supported; other combinations are not supported.
2. **Broadcasting constraints**:
   - The y tensor **does not support broadcasting**. If a dimension of y is 1 while the corresponding dimension of x is not 1, an error will be reported.
   - The x tensor **can be broadcast** to the shape of y.
3. **Shape constraints**: y and x must have the same number of dimensions (1 to 4).
4. **Format constraints**: y and x must have the same format.
5. **In-place update note**: Since AXPY is an in-place operation, the original value of y will be overwritten. To retain the original y value, clone it in advance.

   ```python
   y_backup = pypto.clone(y)  # Back up the original y value.
   y.axpy_(x, alpha=2.0)      # y is updated in place.
   # In this case, y_backup still retains the original value and can be used for subsequent computation.
   ```

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through `set_vec_tile_shapes`.

TileShape must have the same dimensions as the output.

For example, if the input y shape is `[m, n]`, the x shape is `[m, n]` (or `[m, 1]` in a broadcast scenario), the output shape is `[m, n]`, and TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to tile the output `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(32, 32)
```

### API Call Example

#### Basic Usage

```python
y = pypto.tensor([1, 3], pypto.DT_FP32)
x = pypto.tensor([1, 3], pypto.DT_FP32)
y.axpy_(x, alpha=2.0)
```

Result example:

```python
Input data y:   [[1.0 2.0 3.0]]
Input data x:   [[2.0 3.0 4.0]]
alpha:        2.0
Output data y:   [[5.0 8.0 11.0]]  # y = 2.0 * x + y
```

#### Broadcast Scenario

```python
y = pypto.tensor([64, 64], pypto.DT_FP32)  # y shape: [64, 64]
x = pypto.tensor([64, 1], pypto.DT_FP32)   # x shape: [64, 1] (broadcast to [64, 64])
y.axpy_(x, alpha=1.5)
```

#### Retaining the Original y Value

```python
y = pypto.tensor([32, 32], pypto.DT_FP32)
x = pypto.tensor([32, 32], pypto.DT_FP32)

# Back up the original y value in advance if needed.
y_backup = pypto.clone(y)

# Executing AXPY updates y in place.
y.axpy_(x, alpha=2.0)

# y_backup still retains the original value and can be used for other computation.
diff = pypto.sub(y, y_backup)  # Compute the difference between y and the original value.
```

#### Mixed Precision (FP32 + FP16)

```python
y = pypto.tensor([32, 32], pypto.DT_FP32)  # y is FP32.
x = pypto.tensor([32, 32], pypto.DT_FP16)  # x is FP16.
y.axpy_(x, alpha=1.0)  # FP32 (y) + FP16 (x) is supported.
```

#### One-dimensional Scenario

```python
y = pypto.tensor([128], pypto.DT_FP32)
x = pypto.tensor([128], pypto.DT_FP32)
pypto.set_vec_tile_shapes(64)
y.axpy_(x, alpha=2.0)
```
