# pypto.function

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

Defines a PyPTO computation function. Operations required for constructing a computational graph can be added to this function.

## Prototype

```python
function(name: str, *args, **kwargs) -> Iterator
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| name   | Input     | Name of the function, which is used to identify the computational graph.|
| *args  | Input     | Receives the input tensor parameters.|

## Return Value

Returns a context manager, which is used in a with statement.

## Constraints

None

## Examples

```python
with pypto.function("main", a, b, c):
    pypto.set_vec_tile_shapes(16, 16)
    for _ in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        c[:] = a + b
```
