# pypto.get\_conv\_tile\_shapes

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

Obtains the TileShape size configured for convolution (conv) computation and whether L0TileInfo is enabled.

## Prototype

```python
get_conv_tile_shapes() -> Tuple[pypto_impl.TileL1Info, pypto_impl.TileL0Info, bool]
```

## Parameters

None

## Return Value

Returns the TileShape size in the L0 Buffer and L1 Buffer, and whether L0TileInfo is enabled.

## Constraints

None

## Examples

```python
pypto.get_conv_tile_shapes()
```
