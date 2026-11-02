# pypto.get\_cube\_tile\_shapes

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:24:17.672Z pushedAt=2026-10-28T11:57:29.552Z nl=2 -->

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

Obtains the TileShape size configured in cube computation and whether the multi-core K-splitting feature is enabled.

## Prototype

```python
get_cube_tile_shapes() -> Tuple[List[int], List[int], List[int], bool]
```

## Parameters

None

## Return Value

Returns the **TileShape** sizes in the m, k, and n directions, and whether the multi-core K-splitting feature is enabled.

## Constraints

None

## Examples

```python
pypto.get_cube_tile_shapes()
```
