# pypto_pro.language.astype

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:16:40.344Z pushedAt=2026-10-28T11:57:29.508Z nl=2 -->

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

Converts the result of a scalar expression to the specified data type.

## Prototype

```python
result = pypto_pro.language.astype(x, dtype, /)
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| x | Input | Source scalar expression, of the Scalar type, passed by position. |
| dtype | Input | Target data type, of the DataType type, passed by position. |

## Constraints

None

## Return Value

Returns a Scalar of the data type dtype.

## Examples

```python
import pypto_pro.language as pl

# Convert an INT32 scalar to INT64.
source = pl.const(1, pl.DT_INT32)
converted = pl.astype(source, pl.DT_INT64)

# Explicitly unify the scalar types of the two branches of a control flow.
if flag:
    value = pl.astype(lhs, pl.DT_INT64)
else:
    value = pl.astype(rhs, pl.DT_INT64)
```
