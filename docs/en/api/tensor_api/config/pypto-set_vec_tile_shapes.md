# pypto.set\_vec\_tile\_shapes

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

Sets the TileShape size in vector computation.

## Prototype

```python
set_vec_tile_shapes(*args: int) -> None
```

## Parameters

| Parameter | Input/Output| Description                 |
|---------|-----------|-----------------------|
| *args   | Input     | TileShape size of each dimension. A maximum of four inputs are allowed.|

## Return Value

void

## Constraints

TileShape must meet the following constraints:

The size of each dimension must be greater than 0.

Assume that TileShape is two-dimensional \{m, n\}:

- (m \> 0) && \(n \> 0\)

## Examples

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)
```
