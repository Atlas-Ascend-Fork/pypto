# TileOpFormat

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:51:28.179Z pushedAt=2026-10-28T11:57:29.699Z nl=2 -->

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

**TileOpFormat** defines the tile operation format of a tensor, which is used to optimize memory access and computation efficiency in different computation modes. It mainly distinguishes between tile formats for dense computation and sparse computation.

## Prototype

```python
class TileOpFormat(enum.Enum):
     TILEOP_ND = ...  # N-dimensional Tensor that supports standard multi-dimensional array operations.
     TILEOP_NZ = ...  # Same as FRACTAL_NZ/NZ, a format obtained by padding, reshaping, and transposing the lowest two dimensions of a tensor (of all dimensions of a tensor, the right side is the low dimension and the left side is the high dimension).
```
