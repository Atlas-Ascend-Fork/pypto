# pypto.deinterleave

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:22:46.988Z pushedAt=2026-10-28T11:57:29.843Z nl=2 -->

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

Deinterleaves interleaved data into two output tensors. The first output tensor receives the elements at even positions of the interleaved stream, and the second output tensor receives the elements at odd positions of the interleaved stream.

`pypto.deinterleave` supports two input forms:

- **Dual-input form**: `pypto.deinterleave(input, other)`. `input` holds the first half of the interleaved stream, and `other` holds the second half. The API first reconstructs the complete interleaved stream, then splits it by even and odd positions.
- **Single-input form**: `pypto.deinterleave(input)`. The `input` holds the complete interleaved stream. The API splits the data directly along the last dimension by taking even and odd positions separately. The last dimension of the output tensor is half the size of the last dimension of the input tensor.

`pypto.deinterleave` is the inverse operation of `pypto.interleave`.

## Mathematical Semantics

### Dual-Input Form

Given `input` and `other`, first concatenate them along the last dimension into an interleaved stream:

```text
combined[j] = input[j],                         0 <= j < cols
combined[j] = other[j - cols],                  cols <= j < 2 * cols
```

Then deinterleave:

```text
out0[k] = combined[2 * k]
out1[k] = combined[2 * k + 1]
```

### Single-Input Form

Given an `input` that holds the complete interleaved stream:

```text
out0[k] = input[2 * k]
out1[k] = input[2 * k + 1]
```

## Prototype

```python
deinterleave(input: Tensor, other: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]
```

## Parameters

| Parameter | Input/Output | Description |
| ------ | --------- | ---- |
| input  | Input      | Source operand. In the dual-input form, it represents the first half of the interleaved stream; in the single-input form, it represents the complete interleaved stream. The supported type is Tensor. |
| other  | Input      | Optional parameter. In the dual-input form, it represents the second half of the interleaved stream, and its shape and data type must be consistent with `input`. If it is not passed, the single-input form is used. |

## Return Value

Returns a two-tuple `(out0, out1)`.

- `out0` holds the elements at even positions of the interleaved stream.
- `out1` holds the elements at odd positions of the interleaved stream.
- In the dual-input form, the shape and data type of `out0` and `out1` are consistent with the input tensor.
- In the single-input form, the data type of `out0` and `out1` is consistent with `input`; except for the last dimension, the shape is consistent with `input`, and the last dimension is `input.shape[-1] / 2`.

## Constraints

1. The supported data types are: `DT_INT8`, `DT_UINT8`, `DT_INT16`, `DT_UINT16`, `DT_INT32`, `DT_UINT32`, `DT_FP16`, `DT_FP32`, and `DT_BF16`.
2. Tensors of 1 to 4 dimensions are currently supported.
3. In the dual-input form, `input` and `other` must have the same data type, number of dimensions, and shape, and the shape of the last dimension must be an even number.
4. In the single-input form, the shape of the last dimension of `input` must be an even number.
5. When the TileShape is validly configured, its dimensions must be consistent with those of the input tensor.
6. In the dual-input form, the last dimension of the TileShape must be equal to the last dimension of the input tensor Shape, meaning that the last dimension cannot be tiled; other dimensions can be set as required by the tiling requirements.
7. In the single-input form, the last dimension of the TileShape can be tiled, but it must be an even number to ensure that the elements at even and odd positions within each tile are paired; other dimensions can be set as required by the tiling requirements.
8. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

In the dual-input form, the shapes of inputs `input` and `other` and outputs `out0` and `out1` are all `[m, n]`. If the TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to tile the `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

In the single-input form, the shape of `input` is `[m, 2 * n]`, and the shapes of outputs `out0` and `out1` are both `[m, n]`. The last dimension of the TileShape can be smaller than the last dimension of the input tensor. For example, the following configuration is executed tile by tile along the last dimension:

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Dual-Input API Call Example

```python
input = pypto.tensor([2, 4], pypto.DT_FP32)
other = pypto.tensor([2, 4], pypto.DT_FP32)
out0, out1 = pypto.deinterleave(input, other)
```

The results are as follows:

```python
Input data input: [[0.0, 10.0, 1.0, 11.0],
               [4.0, 14.0, 5.0, 15.0]]
Input data other: [[2.0, 12.0, 3.0, 13.0],
               [6.0, 16.0, 7.0, 17.0]]

Output data out0: [[0.0, 1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0, 7.0]]
Output data out1: [[10.0, 11.0, 12.0, 13.0],
               [14.0, 15.0, 16.0, 17.0]]
```

### Single-Input API Call Example

```python
input = pypto.tensor([2, 8], pypto.DT_FP32)
out0, out1 = pypto.deinterleave(input)
```

The results are as follows:

```python
Input data input: [[0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0],
               [4.0, 14.0, 5.0, 15.0, 6.0, 16.0, 7.0, 17.0]]

Output data out0: [[0.0, 1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0, 7.0]]
Output data out1: [[10.0, 11.0, 12.0, 13.0],
               [14.0, 15.0, 16.0, 17.0]]
```

## Related APIs

- [pypto.interleave](pypto-interleave.md): Interleaves two tensors into alternating even/odd element streams and splits them into two output tensors.
