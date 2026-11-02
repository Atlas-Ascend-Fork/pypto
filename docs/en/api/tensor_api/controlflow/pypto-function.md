# pypto.function

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:40:37.659Z pushedAt=2026-10-28T11:57:29.630Z nl=2 -->

## Applicable Products

- Ascend 950PR&950DT products: Supported
- Atlas A3 products: Supported
- Atlas A2 products: Supported

## Description

Defines a PyPTO computation function. Operations required for building a computation graph can be added within this function.

## Prototype

```python
function(name: str, *args, **kwargs) -> Iterator
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| name   | Input      | Name of the function, used to identify the computation graph. |
| *args  | Input      | Used to receive the passed Tensor parameters. |

## Return Value

Returns a context manager, which is used in a **with** statement.

## Constraints

None

## Examples

```python
with pypto.function("main", a, b, c):
    pypto.set_vec_tile_shapes(16, 16)
    for _ in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        c[:] = a + b
```
