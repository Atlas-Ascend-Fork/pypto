# pypto.gathermask

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

Based on the built-in mask selected by PatternMode, collects elements at positions where the corresponding bit of the input tensor is 1 to form the output tensor, and directly discards values at positions where the bit is 0. PatternMode has seven modes:

- PatternMode=1: For the last axis, take the first element of every two elements.
- PatternMode=2: For the last axis, take the second element of every two elements.
- PatternMode=3: For the last axis, take the first element of every four elements.
- PatternMode=4: For the last axis, take the second element of every four elements.
- PatternMode=5: For the last axis, take the third element of every four elements.
- PatternMode=6: For the last axis, take the fourth element of every four elements.
- PatternMode=7: For the last axis, take all elements.

## Prototype

```python
gathermask(self: Tensor, pattern_mode: int) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| self   | Input     | Source operand.<br>Supported type: Tensor.<br>The tensor supports the following data types: DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports 1 to 4 dimensions, and the shape size must not exceed 2147483647 (INT32_MAX).|
| pattern_mode | Input     | Control parameter.<br>Type: int, with a value range of 1 to 7.|

## Return Value

Returns an output tensor. Its data type is the same as that of self, and its shape is as follows:

- If pattern_mode <= 2, the last axis of the output shape is half of the last axis of self.shape, and the other axes are the same as those of the input shape.
- If 2 < pattern_mode < 7, the last axis of the output shape is one quarter of the last axis of self.shape, and the other axes are the same as those of the input shape.
- If pattern_mode = 7, the output shape is the same as the input shape.

## Constraints

1. When 1 <= pattern_mode <= 2:
   - The last axis of self.shape must be divisible by 2.
   - The last axis of TileShape must be a multiple of 2.
   - The last axis of viewshape must be a multiple of 2.
   - The last axis of self.shape is not tiled by view.
2. When 3 <= pattern_mode <= 6:
   - The last axis of self.shape must be a multiple of 4.
   - The last axis of TileShape must be a multiple of 4.
   - The last axis of viewshape must be a multiple of 4.
   - The last axis of self.shape is not tiled by view.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input self is [x, y, z], pattern_mode is 1, the output is [x, y, z/2], and TileShape is set to [x1, y1, 2*z1], then x1, y1, and 2*z1 are used to tile the x, y, and z axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([3, 6], pypto.DT_INT32)        # shape (3, 6)
pattern_mode = 1
y = pypto.gathermask(x, pattern_mode)
```

Result example:

```python
Input data x: [[0,  1,  2,  3,  4,  5],
             [6,  7,  8,  9,  10,  11],
             [12,  13,  14,  15,  16,  17]]
     pattern_mode: 1
Output data y: [[0,  2,  4],
             [6,  8,  10],
             [12,  14,  16]]
```
