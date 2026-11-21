# pypto.get_convbp_filter_tile_shapes

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->

## Description

Obtains the TileShape size configured for conv backward filter computation.

## Prototype

```python
get_convbp_filter_tile_shapes() -> Tuple[pypto_impl.ConvBpTileL1Info, pypto_impl.ConvBpTileL0Info]
```

## Parameters

None

## Return Value

Returns the TileShape size in the L0 Buffer and L1 Buffer.

## Constraints

None

## Examples

```python
pypto.get_convbp_filter_tile_shapes()
```
