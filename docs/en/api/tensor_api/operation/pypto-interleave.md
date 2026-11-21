# pypto.interleave

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

Interleaves the two input tensors (`input` and `other`) element-wise along the last dimension, and tiles the interleaved data stream at the midpoint into two output tensors.

For the elements of the last dimension in each row, an interleaved stream is first constructed as follows:

```text
interleaved[2 * k]     = input[k]
interleaved[2 * k + 1] = other[k]
```

The first half of the interleaved stream is then written to the first output tensor, and the second half is written to the second output tensor. `pypto.interleave` is the inverse operation of `pypto.deinterleave`.

## Prototype

```python
interleave(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]
```

## Parameters

| Parameter| Input/Output| Description|
| ------ | --------- | ---- |
| input  | Input     | First source operand. Supported type: Tensor. Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other  | Input     | Second source operand. Supported type: Tensor. The shape and data type of `other` must be the same as those of `input`. Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a 2-tuple `(out0, out1)`.

- `out0` holds the first half of the interleaved stream.
- `out1` holds the second half of the interleaved stream.
- The shape and data type of `out0` and `out1` are the same as those of `input`.

## Constraints

1. The data type, number of dimensions, and shape of `input` and `other` must be the same.
2. The supported data types are `DT_INT8`, `DT_UINT8`, `DT_INT16`, `DT_UINT16`, `DT_INT32`, `DT_UINT32`, `DT_FP16`, `DT_FP32`, and `DT_BF16`.
3. Currently, 1D to 4D tensors are supported.
4. The shape of the last dimension must be an even number.
5. When TileShape is validly configured, its dimensions must be the same as those of the input tensor, and its last dimension must equal the last dimension of the input tensor's Shape; the other dimensions can be set according to tiling requirements.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

If the shape of inputs `input` and `other` is `[m, n]`, the shape of outputs `out0` and `out1` is `[m, n]`, and TileShape is set to `[m1, n1]`, then `m1` and `n1` are used to tile the `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
input = pypto.tensor([2, 4], pypto.DT_FP32)
other = pypto.tensor([2, 4], pypto.DT_FP32)
out0, out1 = pypto.interleave(input, other)
```

Result example:

```python
Input data input: [[0.0, 1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0, 7.0]]
Input data other: [[10.0, 11.0, 12.0, 13.0],
               [14.0, 15.0, 16.0, 17.0]]

Output data out0: [[0.0, 10.0, 1.0, 11.0],
               [4.0, 14.0, 5.0, 15.0]]
Output data out1: [[2.0, 12.0, 3.0, 13.0],
               [6.0, 16.0, 7.0, 17.0]]
```

## Related APIs

- [pypto.deinterleave](pypto-deinterleave.md): Deinterleaves the interleaved streams back to the element streams at even and odd positions.
