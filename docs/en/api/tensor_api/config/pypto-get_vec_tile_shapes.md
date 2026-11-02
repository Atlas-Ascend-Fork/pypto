# pypto.get\_vec\_tile\_shapes

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:28:11.005Z pushedAt=2026-10-28T11:57:29.575Z nl=2 -->

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

Obtains the **TileShape** size in vector computation.

## Prototype

```python
get_vec_tile_shapes() -> List[int]
```

## Parameters

void

## Return Value

Returns the per-dimension **TileShape** size.

## Constraints

None

## Examples

```python
pypto.get_vec_tile_shapes()
```
