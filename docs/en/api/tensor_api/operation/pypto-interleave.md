# pypto.interleave

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:53:20.754Z pushedAt=2026-10-28T11:57:29.959Z nl=2 -->

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

Interleaves the two input tensors (`input` and `other`) element by element along the last dimension, and splits the interleaved data stream at the midpoint into two output tensors.

For the elements of the last dimension in each row, an interleaved stream is first constructed:

```text
interleaved[2 * k]     = input[k]
interleaved[2 * k + 1] = other[k]
```

The first half of the interleaved stream is then written to the first output tensor, and the second half to the second output tensor. `pypto.interleave` is the inverse operation of `pypto.deinterleave`.

## Prototype

```python
interleave(input: Tensor, other: Tensor) -> Tuple[Tensor, Tensor]
```

## Parameters

| Parameter | Input/Output | Description |
| ------ | --------- | ---- |
| input  | Input      | First source operand. The supported type is Tensor. |
| other  | Input      | Second source operand. The supported type is Tensor. The Shape and data type of `other` must be consistent with those of `input`. |

## Return Value

Returns a two-tuple `(out0, out1)`.

- `out0` stores the first half of the interleaved stream.
- `out1` stores the second half of the interleaved stream.
- The Shape and data type of `out0` and `out1` are consistent with those of `input`.

## Constraints

1. The data type, number of dimensions, and Shape of `input` and `other` must be consistent.
2. The supported data types are: `DT_INT8`, `DT_UINT8`, `DT_INT16`, `DT_UINT16`, `DT_INT32`, `DT_UINT32`, `DT_FP16`, `DT_FP32`, and `DT_BF16`.
3. Tensors of 1 to 4 dimensions are currently supported.
4. The Shape of the last dimension must be an even number.
5. When **TileShape** is validly configured, its dimensions must match those of the input **Tensor**, and its last dimension must equal the last dimension of the input **Tensor**'s **Shape**; the other dimensions can be set according to tiling requirements.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Given that the **Shape** of inputs `input` and `other` is `[m, n]`, and the **Shape** of outputs `out0` and `out1` is `[m, n]`, when **TileShape** is set to `[m1, n1]`, `m1` and `n1` are used to tile the `m` and `n` axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
input = pypto.tensor([2, 4], pypto.DT_FP32)
other = pypto.tensor([2, 4], pypto.DT_FP32)
out0, out1 = pypto.interleave(input, other)
```

The results are as follows:

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

- [pypto.deinterleave](pypto-deinterleave.md): Deinterleaves the interleaved stream back into the even-position and odd-position element streams.
