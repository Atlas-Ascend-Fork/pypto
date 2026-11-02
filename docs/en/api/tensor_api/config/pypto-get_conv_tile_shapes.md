# pypto.get\_conv\_tile\_shapes

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:21:40.219Z pushedAt=2026-10-28T11:57:29.545Z nl=2 -->

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

Obtains the TileShape size configured for convolution (conv) computation and whether L0TileInfo is enabled.

## Prototype

```python
get_conv_tile_shapes() -> Tuple[pypto_impl.TileL1Info, pypto_impl.TileL0Info, bool]
```

## Parameters

None

## Return Value

Returns the **TileShape** sizes on L0 and L1, and whether the **L0TileInfo** switch is enabled.

## Constraints

None

## Examples

```python
pypto.get_conv_tile_shapes()
```
