# pypto.pad

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
<!-- npu="9030" id5 -->
- Kirin 9030 processors: Supported
<!-- end id5 -->
<!-- npu="x90" id4 -->
- Kirin X90 processors: Supported
<!-- end id4 -->
<!-- npu="9050" id8 -->
- Kirin 9050 processors: Supported
<!-- end id8 -->

## Description

Pads the input tensor.

The padding size is described from the last dimension of the input tensor backwards based on the `pad` parameter. The `pad` parameter is in the format of $(pad\_left, pad\_right, pad\_top, pad\_bottom, ...)$. The current implementation only supports constant padding on the right and bottom of the last two dimensions.

## Prototype

```python
pad(input: Tensor, pad: Sequence[int], mode: str = "constant", value: Union[float, int] = 0) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                                                                                                                                                                          |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | Input     | Source operand to be padded.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, and DT_UINT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX). <!-- npu="x90,9030,9050" id6 -->Kirin processors support the following data types: DT_FP32 and DT_FP16.<!-- end id6 -->                                                  |
| pad    | Input     | Padding size sequence.<br>Supported types: tuple or list (containing int).<br>The sequence length $m$ must be an even number and satisfy $\frac{m}{2} \leq$ the number of dimensions of `input`.<br>Format: `(pad_left, pad_right, pad_top, pad_bottom, ...)`<br>All values in the padding size sequence must be non-negative integers. Negative values are not supported.                          |
| mode   | Input     | Padding mode.<br>Supported type: str.<br>The value can be `'constant'`, `'reflect'`, `'replicate'`, or `'circular'`.<br>Default value: `'constant'`.<br>**Note**: Currently, only the `'constant'` mode is supported.                                            |
| value  | Input     | Padding value when the padding mode is `'constant'`.<br>Supported types: float and int.<br>For floating-point types (DT_FP32, DT_FP16, and DT_BF16), any floating-point value is supported, including `-inf`, `inf`, `0.0`, and any other floating-point number (such as `1.0`, `-1.0` and `0.5`).<br>For integer types (DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, and DT_UINT32), **only integer values are supported**. Passing `float('-inf')` or `float('inf')` is not supported. If you do so, PyPTO will raise a `ValueError` to prompt you to pass an actual integer value.<br>Default value: `0`.                                                                                                                                         |

## Return Value

Returns the output tensor. Its data type is the same as that of `input`, and its shape is the size expanded on the corresponding dimensions according to the `pad` parameter.

## Constraints

1. The length of the `pad` parameter must be 2 or 4, and all values in the padding size sequence of `pad` must be non-negative integers. Negative padding is not supported. If a negative value is passed, a `ValueError` is raised.
2. Currently, **only right and bottom padding in the multi-dimensional case, or right padding in the 1-dimensional case, is supported**. <!-- npu="x90,9030,9050" id7 -->Kirin processors additionally support left and top padding, meaning padding on all four sides is supported. <!-- end id7 -->That is, the leftward and upward padding amounts in the `pad` sequence must be 0 (for example, the format must be `(0, pad_right, 0, pad_bottom)` or `(0, pad_right)`).
3. Currently, **only the `'constant'` mode is supported**.
4. **Integer types do not support floating-point values**: For integer dtypes (DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, and DT_UINT32), passing a floating-point value to `value` is not supported. If you do so, PyPTO will raise a `ValueError`. To pad with the minimum or maximum value of an integer type, explicitly pass the actual value of the corresponding dtype (for example, `-2147483648` for `int32` and `-32768` for `int16`).
5. If `input` is not of the Tensor type or `pad` is not an integer sequence, a `TypeError` is raised.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through `set_vec_tile_shapes`.

TileShape must have the same dimensions as the **output**.

Example: If the input shape is [1, 1, 2, 2], 1 is padded on the right of the last axis and at the bottom of the second-to-last axis, and the output is [1, 1, 3, 3], TileShape can be set to [1, 1, 4, 16].

```python
pypto.set_vec_tile_shapes(1, 1, 4, 16)
```

### API Call Example

The following is the API call snippet in the computational graph construction context. The first parameter of pypto.tensor is Shape, which does not initialize the example value. The specific input data is provided by the caller during execution.

```python
# Example: Pad a tensor with shape [1, 1, 2, 2].
# Pad 1 on the last dimension (right side).
# Pad 1 on the second-to-last dimension (bottom).
t4d = pypto.tensor([1, 1, 2, 2], pypto.DT_FP32)

p1 = (0, 1, 0, 1)  # (pad_left=0, pad_right=1, pad_top=0, pad_bottom=1)
out = pypto.pad(t4d, p1, mode="constant", value=0.0)
```

When the caller provides the following input data, the result is as follows:

```python
# Input data t4d (logical shape [1, 1, 2, 2]):
[[[[0.0, 1.0],
   [2.0, 3.0]]]]

# Output data out (logical shape expanded to [1, 1, 3, 3]):
[[[[0.0, 1.0, 0.0],
    [2.0, 3.0, 0.0],
    [0.0, 0.0, 0.0]]]]
```
