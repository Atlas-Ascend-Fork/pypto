# pypto_pro.language.make_tuple

<!-- md-trans-meta sourceCommit=d15a7740ea679c63c4ade0c9eaf473865b3b634e translatedAt=2026-10-26T14:53:38.121Z pushedAt=2026-10-28T11:57:29.331Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Creates a compile-time named tuple that aggregates multiple IR variables by field name. Field access is resolved to the corresponding original value at compile time, without generating a C++ struct or incurring runtime overhead.
[pypto_pro.language.struct](./struct.md) generates a C++ struct and can be used to pass data across Pipelines. The two differ.

## Prototype

```python
pypto_pro.language.make_tuple(**kwargs: Any) -> Any
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| \*\*kwargs | Input | Named fields, passed in the form field=value, with at least one field specified. Field names must be valid identifiers. Field values support value expressions resolvable within the Kernel, including Python scalar values, Kernel scalar expressions, Tensor, Tile, Ptr, and Python tuples composed of these values, such as (tile0, tile1). Positional arguments are not supported, nor is expanding arguments through a dictionary, such as make_tuple(\*\*fields). |

## Constraints

The returned named tuple object supports accessing the packed variables by field name. Field access is constant-folded back to the original value at compile time, incurring no runtime overhead.

### Use Cases

- When a function returns multiple values, you can aggregate multiple Tiles or variables and return them.
- You can aggregate logically related variables and access them by field name.
- You can aggregate multiple Tiles in a ping-pong buffer and access them by field name.

### Differences from pypto_pro.language.struct

- Use pypto_pro.language.struct when data needs to be passed across pipelines, for example, when communicating through SSBUF.
- Use pypto_pro.language.make_tuple when variables only need to be aggregated within the same pipeline.

## Return Value

Returns a named tuple object containing all named fields.

## Examples

### Create a Named Tuple and Access Fields

```python
import pypto_pro.language as pl

# Scenario 1: Scalar field packing + field access
@pl.jit()
def tuple_scalar_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TScalar", a=11, b=22)
    with pl.section_vector():
        t = pl.make_tuple(first=s.a, second=s.b)
        pl.setval(out, 0, t.first + t.second)
        pl.setval(out, 1, t.second - t.first)
# Scenario 2: Packing inside a for loop + struct field relay
@pl.jit()
def tuple_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("LoopT", v=0, cur=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            acc.cur = i
            t = pl.make_tuple(x=acc.cur, y=acc.cur * 10)
            acc.v = acc.v + t.x + t.y
        pl.setval(out, 0, acc.v)
```
