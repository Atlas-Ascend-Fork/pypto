# pypto.set\_vec\_tile\_shapes

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:36:37.677Z pushedAt=2026-10-28T11:57:29.617Z nl=2 -->

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

Sets the TileShape size in vector computation.

## Prototype

```python
set_vec_tile_shapes(*args: int) -> None
```

## Parameters

| Parameter  | Input/Output | Description                  |
|---------|-----------|-----------------------|
| *args   | Input      | TileShape size of each dimension, with a maximum of 4 inputs |

## Return Value

void

## Constraints

TileShape must meet the following constraints:

Each dimension must be greater than 0.

Assume that TileShape is two-dimensional \{m, n\}; then:

- (m \> 0) && \(n \> 0\)

## Examples

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)
```
