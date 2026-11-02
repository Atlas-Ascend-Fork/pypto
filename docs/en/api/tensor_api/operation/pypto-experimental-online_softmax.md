# pypto.experimental.online\_softmax

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:34:59.727Z pushedAt=2026-10-28T11:57:29.893Z nl=2 -->

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

This operator performs block-wise online Softmax computation and computes local statistics on the input **scores** along dimension 0. The operator first multiplies **scores** by **scale**, then computes the maximum value and the exponential sum of each column, and outputs the unnormalized exponential results. This API is typically used in block-wise attention scenarios such as FlashAttention, in conjunction with `pypto.experimental.online_softmax_update` to update the global maximum, exponential sum, and intermediate outputs block by block.

## Prototype

```python
online_softmax(scores: Tensor, scale: float) -> Tuple[Tensor, Tensor, Tensor]
```

## Parameters

| Parameter | Input/Output | Description |
|--------|-----------|------|
| scores | Input | Source operand.<br>Supported data type: DT_FP32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>The shape is [k_len, q_len]. |
| scale | Input | float type.<br>Scalar used to scale **scores**, commonly set to `1.0 / sqrt(head_dim)`. |

## Return Value

Returns three output tensors:

| Return Value | Description |
|--------|------|
| exp_scores_bf16 | Exponential results after scaling and subtracting the column maximum, with data type DT_BF16 and the same shape as **scores**. |
| column_max | Local maximum of each column, with data type DT_FP32 and shape [1, q_len]. |
| column_sum | Local exponential sum of each column, with data type DT_FP32 and shape [1, q_len]. |

## Constraints

1. This API is a custom API, and its stability is not guaranteed.
2. The data type of **scores** supports only DT_FP32.
3. In the current version, dimension 0 is not tiled. The constraint `scores.shape[0] <= vec_tile[0]` must be satisfied.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through `set_vec_tile_shapes`.

The dimension settings of the TileShape must be consistent with the input **scores**. In the current version, dimension 0 is not tiled. The constraint `scores.shape[0] <= vec_tile[0]` must be satisfied.

### API Call Example

```python
import pypto

scores = pypto.tensor([128, 128], pypto.DT_FP32)
scale = 1.0 / (128 ** 0.5)

pypto.set_vec_tile_shapes(128, 64)
exp_scores_bf16, column_max, column_sum = pypto.experimental.online_softmax(scores, scale)
```
