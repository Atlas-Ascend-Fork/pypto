# TileOpFormat

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
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

TileOpFormat defines the tile operation format of a tensor. It is used to optimize the memory access and computation efficiency in different computation modes. It mainly distinguishes between tile formats for dense computation and sparse computation.

## Prototype

```python
class TileOpFormat(enum.Enum):
     TILEOP_ND = ...  # N-dimensional tensor, supporting standard multi-dimensional array operations.
     TILEOP_NZ = ...  # Same as FRACTAL_NZ/NZ, a format obtained after the two lowest dimensions of a tensor are padded, reshaped, and transposed (of all dimensions of a tensor, low dimensions are on the right side, and high dimensions are on the left side).
```
