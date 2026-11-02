# pypto.axpy_

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:07:08.194Z pushedAt=2026-10-28T11:57:29.784Z nl=2 -->

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

Performs the AXPY operation: `y = alpha * x + y`. This operation updates the y tensor in place.

The computation formula is as follows:

$$
y_i = \alpha \cdot x_i + y_i
$$

**Important note**: AXPY is an in-place operation, and the y tensor is modified directly. If the original y value before AXPY is needed for subsequent computation, back it up with `pypto.clone(y)` before calling AXPY.

## Prototype

```python
axpy_(y: Tensor, x: Tensor, alpha: Union[int, float] = 1.0) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| y      | Input/Output | Target tensor, which is updated in place.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32 and DT_FP16.<br>**Broadcast not supported**: The shape of y must accommodate the broadcast result of x, that is, no dimension of y can be 1 (unless the corresponding dimension of x is also 1).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |
| x      | Input      | Source tensor, which can be broadcast to the shape of y.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32 and DT_FP16.<br>Broadcast supported: x can be broadcast to the shape of y (for example, x has shape `[m, 1]` and y has shape `[m, n]`).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**).                                                 |
| alpha  | Input      | Scaling factor used to scale x.<br>Supported types: int and float, with a default value of **1.0**.<br>The data type of alpha is automatically converted to match y.                                                                                                            |

## Return Value

Returns the updated tensor y (which shares the same memory address as the input y), with the same data type and shape as y.

## Constraints

1. **dtype constraint**:
   - Same dtype: DT_FP32 + DT_FP32 and DT_FP16 + DT_FP16 are supported.
   - Mixed dtype: only DT_FP32 (y) + DT_FP16 (x) is supported; other combinations are not supported.
2. **Broadcast constraints**:
   - The y tensor **does not support broadcasting**. If a dimension of y is 1 while the corresponding dimension of x is not 1, an error is reported.
   - The x tensor **supports broadcasting** to the shape of y.
3. **Shape constraints**: The number of dimensions of y and x must be the same (1 to 4 dimensions).
4. **Format constraints**: The format of y and x must be consistent.
5. **In-place update note**: Since AXPY is an in-place operation, the original value of y will be overwritten. To retain the original y value, clone it in advance:

   ```python
   y_backup = pypto.clone(y)  # Back up the original y value.
   y.axpy_(x, alpha=2.0)      # y is updated in place.
   # At this point, y_backup still retains the original value and can be used for subsequent computation.
   ```

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through `set_vec_tile_shapes`.

The TileShape must have the same number of dimensions as the output.

Example: The input y shape is `[m, n]`, the x shape is `[m, n]` (or `[m, 1]` in the broadcast scenario), the output shape is `[m, n]`, and **TileShape** is set to `[m1, n1]`. Then `m1` and `n1` are used to tile the `m` and `n` axes of the output, respectively.

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

The results are as follows:

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

# Execute AXPY. y is updated in place.
y.axpy_(x, alpha=2.0)

# y_backup still retains the original value and can be used for other computations.
diff = pypto.sub(y, y_backup)  # Compute the difference between y and the original value.
```

#### Mixed Precision (FP32 + FP16)

```python
y = pypto.tensor([32, 32], pypto.DT_FP32)  # y is FP32.
x = pypto.tensor([32, 32], pypto.DT_FP16)  # x is FP16.
y.axpy_(x, alpha=1.0)  # Supports FP32(y) + FP16(x).
```

#### 1D Scenario

```python
y = pypto.tensor([128], pypto.DT_FP32)
x = pypto.tensor([128], pypto.DT_FP32)
pypto.set_vec_tile_shapes(64)
y.axpy_(x, alpha=2.0)
```
