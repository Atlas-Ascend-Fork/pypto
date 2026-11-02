# pypto.get_convbp_input_tile_shapes

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:22:21.628Z pushedAt=2026-10-28T11:57:29.549Z nl=2 -->

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

Obtains the TileShape size set for the conv backward input computation.

## Prototype

```python
get_convbp_input_tile_shapes() -> Tuple[pypto_impl.ConvBpTileL1Info, pypto_impl.ConvBpTileL0Info]
```

## Parameters

None

## Return Value

Returns the TileShape sizes on L0 and L1.

## Constraints

None

## Examples

```python
pypto.get_convbp_input_tile_shapes()
```
