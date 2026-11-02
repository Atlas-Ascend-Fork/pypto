# pypto_pro.language.eq

<!-- md-trans-meta sourceCommit=099ec62fdd650eea5fa0fd14533d5f0f32da389d translatedAt=2026-10-26T14:37:50.048Z pushedAt=2026-10-28T11:57:29.220Z nl=2 -->

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

Compares lhs and rhs element-wise for equality and generates a bit-packed mask. rhs can be a Tile or a scalar. The data type of the mask Tile is not restricted; DT_UINT8 is commonly used, in conjunction with [pypto_pro.language.select](../selection/select.md).

## Prototype

```python
pypto_pro.language.eq(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of Tile type. The data type is not restricted; DT_UINT8 is commonly used, in conjunction with a bit-packed mask layout. |
| lhs | Input | Source operand (left operand) of Tile type, stored in UB with row-major layout. Supports 8-, 16-, 32-, and 64-bit integers, DT_FP16, DT_BF16, and DT_FP32. |
| rhs | Input | Source operand (right operand) of Tile or Scalar type, also supporting Python int or float constants that can be converted to Scalar. When a Tile is passed, it is stored in UB with row-major layout, and its data type, shape, and valid shape must match lhs. When a Scalar or Python constant is passed, its data type must be compatible with the element type of lhs. |

## Constraints

None

## Return Value

None

## Examples

```python
pl.eq(mask, lhs, rhs)  # Tile-Tile comparison: lhs == rhs
pl.eq(mask, lhs, 0)    # Tile-Scalar comparison: lhs == 0
```
