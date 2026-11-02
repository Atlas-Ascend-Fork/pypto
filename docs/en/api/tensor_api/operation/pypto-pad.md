# pypto.pad

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:14:09.012Z pushedAt=2026-10-28T11:57:30.030Z nl=2 -->

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

Pads the input tensor.

The padding size is described from the last dimension of the input tensor backwards, according to the `pad` parameter. The format of the `pad` parameter is $(pad\_left, pad\_right, pad\_top, pad\_bottom, ...)$. The current implementation supports only constant-mode padding on the right and bottom of the last two dimensions.

## Prototype

```python
pad(input: Tensor, pad: Sequence[int], mode: str = "constant", value: Union[float, int] = 0) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                           |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | Input      | Source operand to be padded.<br>Supported type: Tensor.<br>Supported Tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).                                                  |
| pad    | Input      | Padding size sequence.<br>Supported type: tuple or list (containing int).<br>The sequence length $m$ must be an even number and satisfy $\frac{m}{2} \leq$ the number of dimensions of `input`.<br>Format: `(pad_left, pad_right, pad_top, pad_bottom, ...)`.<br>All values in the padding size sequence must be non-negative integers. Negative values are not supported.                           |
| mode   | Input      | Padding mode.<br>Supported type: str.<br>Optional values: `'constant'`, `'reflect'`, `'replicate'`, or `'circular'`.<br>Default value: `'constant'`.<br>**Note**: Currently only the `'constant'` mode is supported.                                             |
| value  | Input      | Padding value when the padding mode is constant padding (`'constant'`).<br>Supported type: float or int.<br>For floating-point types (DT_FP32, DT_FP16, DT_BF16), any floating-point value is supported, including `-inf`, `inf`, `0.0`, and any other floating-point number (such as `1.0`, `-1.0`, `0.5`, etc.).<br>For integer types (DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32), value **supports only integer values**. Passing `float('-inf')` or `float('inf')` is not supported, and PyPTO raises a `ValueError` to prompt the user to pass an actual integer value.<br>Default value: `0`.                                                                                                                                          |

## Return Value

Returns the output tensor. The data type of the tensor is the same as that of `input`, and its shape is the size expanded on the corresponding dimensions according to the `pad` parameter.

## Constraints

1. The length of the `pad` parameter must be 2 or 4, and all values in the padding size sequence of the `pad` parameter must be non-negative integers. Negative padding is not supported. If a negative value is passed, a `ValueError` is raised.
2. Currently, **only right and bottom padding in the multi-dimensional case, or right padding in the 1-dimensional case, is supported**. That is, the leftward and upward padding amounts in the `pad` sequence must be 0 (for example, the format must be `(0, pad_right, 0, pad_bottom)` or `(0, pad_right)`).
3. Currently, mode **supports only the `'constant'` (constant padding) mode**. Other modes are not supported yet.
4. **Integer types do not support floating-point values**: For integer dtypes (DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32), the `value` parameter does not support passing a floating-point value, and PyPTO raises a `ValueError`. To pad with the minimum or maximum value of an integer type, explicitly pass the actual value of the corresponding dtype (for example, `-2147483648` for `int32` and `-32768` for `int16`).
5. If `input` is not of the Tensor type, or `pad` is not an integer sequence, a `TypeError` is raised.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through `set_vec_tile_shapes`.

The TileShape dimensions must be consistent with the **output**.

Example 1: If the input `input` has shape `[m, n]` and `p` is padded on the right side of the n axis, the output shape is `[m, n+p]`. If the TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to split the `m` and `n+p` axes of the output, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
# Example: Pad a tensor with shape [1, 1, 2, 2].
# Pad 1 on the last dimension (right side).
# Pad 1 on the second-to-last dimension (bottom).
t4d = pypto.tensor([0.0, 1.0, 2.0, 3.0], pypto.DT_FP32)
# Assume the 1D data has been reshaped to [1, 1, 2, 2] internally.

p1 = (0, 1, 0, 1)  # (pad_left=0, pad_right=1, pad_top=0, pad_bottom=1)
out = pypto.pad(t4d, p1, mode="constant", value=0.0)
```

The results are as follows:

```python
# Input data t4d (logical shape [1, 1, 2, 2]):
[[[[0.0, 1.0],
   [2.0, 3.0]]]]

# Output data out (logical shape expanded to [1, 1, 3, 3]):
[[[[0.0, 1.0, 0.0],
    [2.0, 3.0, 0.0],
    [0.0, 0.0, 0.0]]]]
```
