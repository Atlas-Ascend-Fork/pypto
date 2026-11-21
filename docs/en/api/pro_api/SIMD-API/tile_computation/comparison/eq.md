# pypto_pro.language.eq

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

Compares lhs and rhs element-wise to check for equality and generates a bit-packed mask. rhs can be a Tile or a scalar. The data type of the mask Tile is not limited. Generally, the DT_UINT8 type is used together with [pypto_pro.language.select](../selection/select.md).

## Prototype

```python
pypto_pro.language.eq(out: Tile, lhs: Tile, rhs: Union[Tile, int, float]) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, which is of type Tile. The data type is not restricted; DT_UINT8 is typically used, in conjunction with a bit-packed mask layout.|
| lhs | Input| Source operand (left operand), which is of type Tile, stored in Unified Buffer (UB), and laid out in row-major order. Supported data types: 8-bit, 16-bit, 32-bit, and 64-bit integers, DT_FP16, DT_BF16, and DT_FP32.|
| rhs | Input| Source operand (right operand), which can be a Tile, an integer, or a floating-point constant, or a runtime scalar expression of the corresponding type. When a Tile is passed in, the Tile's storage is UB, the Tile uses a row-major layout, and the Tile's data type, shape, and valid shape must match those of lhs. When a non-Tile value is passed in, the non-Tile value's data type must be compatible with lhs's element type.|

## Constraints

None.

## Return Value

None.

## Examples

```python
pl.eq (mask, lhs, rhs)  # Tile-Tile comparison: lhs == rhs
pl.eq (mask, lhs, 0)    # Tile-Scalar comparison: lhs == 0
```
