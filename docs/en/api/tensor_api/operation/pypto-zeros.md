# pypto.zeros

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:48:48.038Z pushedAt=2026-10-28T11:57:30.162Z nl=2 -->

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

Creates a tensor of the specified `size`, with all elements initialized to `0`. The data type is specified by `dtype` and defaults to `DT_FP32` if not provided.

## Precautions

- **TileShape must be set first**: Before calling this API, you must set the TileShape through [set_vec_tile_shapes](../config/pypto-set_vec_tile_shapes.md).
- **The dtype parameter must be passed explicitly**: When a data type needs to be specified, it must be passed explicitly using the keyword argument `dtype=` and cannot be passed as a positional argument. For example, use `pypto.zeros(2, 3, dtype=pypto.DT_INT32)` instead of `pypto.zeros(2, 3, pypto.DT_INT32)`. If passed as a positional argument, the dtype value is mistakenly parsed as a dimension of **size**, causing an error.

## Prototype

```python
zeros(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor
```

## Parameters

| Parameter       | Input/Output | Description                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| *size        | Input      | Source operand, used to define the shape of the output tensor.<br>Supports a variable number of arguments (multiple ints) or a single sequence (such as List[int] or Tuple[int]). |
| dtype        | Input      | Optional parameter used to define the data type of the output tensor.<br>Supported data types: `DT_FP32`, `DT_INT32`, `DT_INT16`, `DT_FP16`, and `DT_BF16`.<br>The default value is `pypto.DT_FP32`. |

## Return Value

Returns an output tensor. Its data type is determined by `dtype`, its shape is the specified `size`, and all elements are `0`.

## Constraints

1. The dimensions of the TileShape must be consistent with those of the output `result`, as it is used to tile the `result`.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through `set_vec_tile_shapes`. The dimensions of the TileShape must be consistent with those of the output.
For example, if the input size is `[m, n]`, the output is `[m, n]`, and the TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to tile the `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(2, 3)
```

### API Call Example

```python
# Example 1: Pass size as variable arguments and use the default dtype (DT_FP32).
x1 = pypto.zeros(2, 3)

# Example 2: Pass size as a tuple and explicitly specify dtype (DT_INT32).
x2 = pypto.zeros((2, 3), dtype=pypto.DT_INT32)
```

The results are as follows:

```python
x1 output data: [[0., 0., 0.],
             [0., 0., 0.]]
x2 output data: [[0, 0, 0],
             [0, 0, 0]]
```
