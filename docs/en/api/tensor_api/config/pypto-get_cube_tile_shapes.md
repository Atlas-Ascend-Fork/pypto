# pypto.get\_cube\_tile\_shapes

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

Obtains the TileShape size configured for cube computation and whether the multi-core K switching feature is enabled.

## Prototype

```python
get_cube_tile_shapes() -> Tuple[List[int], List[int], List[int], bool]
```

## Parameters

None

## Return Value

Returns the TileShape sizes in the m, k, and n directions and whether the multi-core K switching feature is enabled.

## Constraints

None

## Examples

```python
pypto.get_cube_tile_shapes()
```
