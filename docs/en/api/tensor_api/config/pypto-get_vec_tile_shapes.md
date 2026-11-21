# pypto.get\_vec\_tile\_shapes

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

Obtains the TileShape size in vector computation.

## Prototype

```python
get_vec_tile_shapes() -> List[int]
```

## Parameters

void

## Return Value

Returns the TileShape size of each dimension.

## Constraints

None

## Examples

```python
pypto.get_vec_tile_shapes()
```
