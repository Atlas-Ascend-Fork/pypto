# pypto.fillpad

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

Pads the input tensor.

Unlike the pad API, this API does not change the shape of the tensor. It fills the padding region (the region beyond the valid shape) with the specified value. The current implementation supports 1D or 2D input tensors and performs constant padding on the right and bottom.

## Prototype

```python
fillpad(input: Tensor, mode: str = "constant", value: Union[float, int] = 0) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                                                                                                                                                                          |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | Input     | Source operand to be padded.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, and DT_UINT32.<br>Empty tensors are not supported. The shape supports only 1 to 2 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).                                                    |
| mode   | Input     | Padding mode.<br>Supported type: str.<br>The value can be `'constant'`, `'reflect'`, `'replicate'`, or `'circular'`.<br>Default value: `'constant'`.<br>**Note**: Currently, only the `'constant'` mode is supported.                                            |
| value  | Input     | Padding value when the padding mode is `'constant'`.<br>Supported types: float and int.<br>For floating-point types (DT_FP32, DT_FP16, and DT_BF16), any floating-point value is supported, including `-inf`, `inf`, `0.0`, and any other floating-point number (such as `1.0`, `-1.0`, and `0.5`).<br>For integer types (DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, and DT_UINT32), any integer value is supported.<br>Default value: `0`.|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

## Constraints

1. Currently, **only the `'constant'` mode is supported**.
2. The value can be any floating-point or integer value. The data type of the padding value is automatically converted to be the same as that of the input tensor.
3. If `input` is not of the Tensor type, `TypeError` is thrown.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through `set_vec_tile_shapes`.

TileShape must have the same dimensions as the **output**.

Example 1: If the input `input` shape is `[m, n]`, the output shape is `[m, n]`, and TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to tile the `m` and `n` axes of the output, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([4, 4], pypto.DT_FP32)
out = pypto.fillpad(a, "constant", "-inf")
```

Result example:

```python
# Input data t4d (logical shape [4, 4]):
[[1.0, 2.0, 0.0, 0.0],
[3.0, 4.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0]]

# Output data out (logical shape [4, 4]):
[[1.0, 2.0, -inf, -inf],
[3.0, 4.0, -inf, -inf],
[-inf, -inf, -inf, -inf],
[-inf, -inf, -inf, -inf]]
```
