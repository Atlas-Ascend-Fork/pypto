# pypto.experimental.online\_softmax\_update

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:34:23.404Z pushedAt=2026-10-28T11:57:29.886Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

This is a custom API with many constraints. Its stability is not guaranteed.

This operator is used for state update in online Softmax. Given the maximum value, exponential sum, and intermediate output of the historical block, as well as the maximum value, exponential sum, and intermediate output of the current block, the operator merges the two parts of state according to the online Softmax formula to obtain the updated maximum value, exponential sum, and unnormalized output.

This API is typically used together with `pypto.experimental.online_softmax`: `online_softmax` computes the local statistics of the current scores block, and `online_softmax_update` merges the current block's statistics into the existing state. The final output usually still needs to be normalized using the updated exponential sum.

## Prototype

```python
online_softmax_update(
    previous_max: Tensor,
    previous_sum: Tensor,
    previous_output: Tensor,
    current_max: Tensor,
    current_sum: Tensor,
    current_output: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| previous_max | Input | Column maximum of the historical block.<br>Supported data type: DT_FP32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>Shape: [1, q_len]. |
| previous_sum | Input | Column exponential sum of the historical block.<br>Supported data type: DT_FP32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>Shape: [1, q_len]. |
| previous_output | Input | Unnormalized output accumulated by the historical block.<br>Supported data type: DT_FP32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>Shape: [head_dim, q_len]. |
| current_max | Input | Column maximum of the current block, typically from `pypto.experimental.online_softmax`.<br>Supported data type: DT_FP32.<br>Shape: [1, q_len]. |
| current_sum | Input | Column exponential sum of the current block, typically from `pypto.experimental.online_softmax`.<br>Supported data type: DT_FP32.<br>Shape: [1, q_len]. |
| current_output | Input | Unnormalized output of the current block.<br>Supported data type: DT_FP32.<br>Shape: [head_dim, q_len], which must be consistent with the shape of **previous_output**. |

## Return Value

Returns three output tensors:

| Return Value | Description |
|--------|------|
| updated_max | Merged column maximum, with data type DT_FP32 and shape [1, q_len]. |
| updated_sum | Merged column exponential sum, with data type DT_FP32 and shape [1, q_len]. |
| updated_output | Merged unnormalized output, with data type DT_FP32 and shape [head_dim, q_len]. |

## Constraints

1. This API is a custom API, and its stability is not guaranteed.
2. All input tensors support only the DT_FP32 data type.
3. **current_output** must have the same shape as **previous_output**.
4. In the current version, dimension 0 is not tiled. The constraint `previous_output.shape[0] <= vec_tile[0]` must be satisfied.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through `set_vec_tile_shapes`.

The dimension settings of the TileShape must be consistent with `previous_output` and `current_output`. In the current version, dimension 0 is not tiled. The constraint `previous_output.shape[0] <= vec_tile[0]` must be satisfied. In addition, the last dimension of the tile must satisfy the 32-byte alignment requirement for FP32.

### API Call Example

```python
import pypto

previous_max = pypto.tensor([1, 128], pypto.DT_FP32)
previous_sum = pypto.tensor([1, 128], pypto.DT_FP32)
previous_output = pypto.tensor([128, 128], pypto.DT_FP32)
current_max = pypto.tensor([1, 128], pypto.DT_FP32)
current_sum = pypto.tensor([1, 128], pypto.DT_FP32)
current_output = pypto.tensor([128, 128], pypto.DT_FP32)

pypto.set_vec_tile_shapes(128, 64)
updated_max, updated_sum, updated_output = pypto.experimental.online_softmax_update(
    previous_max,
    previous_sum,
    previous_output,
    current_max,
    current_sum,
    current_output,
)
```
