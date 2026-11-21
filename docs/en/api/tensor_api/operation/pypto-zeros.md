# pypto.zeros

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

Creates a tensor of the specified `size` with all elements initialized to `0`. Its data type is specified by `dtype` and defaults to `DT_FP32`.

## Precautions

- **TileShape must be set first**: Before calling this API, you must set TileShape through [set_vec_tile_shapes](../config/pypto-set_vec_tile_shapes.md).
- **dtype must be explicitly passed**: When the data type needs to be specified, it must be passed explicitly using the keyword parameter `dtype=`, and cannot be passed as a positional parameter. For example, use `pypto.zeros(2, 3, dtype=pypto.DT_INT32)` instead of `pypto.zeros(2, 3, pypto.DT_INT32)`. If it is passed as a positional parameter, the dtype value is mistakenly parsed as a dimension of size, causing an error.

## Prototype

```python
zeros(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor
```

## Parameters

| Parameter      | Input/Output| Description                                                                |
|--------------|-----------|----------------------------------------------------------------------|
| *size        | Input     | Source operand that defines the shape of the output tensor.<br>It supports variable-length parameters (multiple ints) or a single sequence (such as List[int] or Tuple[int]).|
| dtype        | Input     | Optional parameter used to define the data type of the output tensor.<br>Supported data types: `DT_FP32`, `DT_INT32`, `DT_INT16`, `DT_FP16`, and `DT_BF16`.<br>The default value is `pypto.DT_FP32`.|

## Return Value

Returns the output tensor. Its data type is determined by `dtype`, its shape is the size specified by `size`, and all elements are `0`.

## Constraints

1. The dimensions of `tileshape` must be the same as those of the output result, which is used to tile the result.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through `set_vec_tile_shapes`. TileShape must have the same dimensions as the output.
For example, if the input size is `[m, n]`, the output size is `[m, n]`, and TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to tile the `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(2, 3)
```

### API Call Example

```python
# Example 1: Pass size as a variable parameter and use the default dtype (DT_FP32).
x1 = pypto.zeros(2, 3)

# Example 2: Pass size as a tuple and explicitly specify dtype (DT_INT32).
x2 = pypto.zeros((2, 3), dtype=pypto.DT_INT32)
```

Result example:

```python
x1 output data: [[0., 0., 0.],
             [0., 0., 0.]]
x2 output data: [[0, 0, 0],
             [0, 0, 0]]
```
