# pypto_pro.language.const

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:17:23.888Z pushedAt=2026-10-28T11:57:29.515Z nl=2 -->

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

Creates a compile-time constant scalar of a specified data type. It is used in scenarios that require a scalar value with an explicitly specified type, such as comparing with a Tile of the pypto_pro.language.DT_INT32 type or initializing an index value.

## Prototype

```python
result = pypto_pro.language.const(value, dtype, /)
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Numeric constant, supporting Python int or float constant values. |
| dtype | Input | Target data type, supporting pypto_pro.language.DT_INT8, pypto_pro.language.DT_INT16, pypto_pro.language.DT_INT32, pypto_pro.language.DT_INT64, pypto_pro.language.DT_UINT8, pypto_pro.language.DT_UINT16, pypto_pro.language.DT_UINT32, pypto_pro.language.DT_UINT64, pypto_pro.language.DT_FP16, pypto_pro.language.DT_FP32, pypto_pro.language.DT_BF16, pypto_pro.language.DT_BOOL, and so on. |

## Constraints

None

## Return Value

A scalar value (Scalar), with the type specified by dtype.

## Examples

```python
import pypto_pro.language as pl
# Create a zero constant of type INT64.
zero_idx = pl.const(0, pl.DT_INT64)

# Create a constant of type INT32.
offset = pl.const(42, pl.DT_INT32)

# Create a constant of type FP16.
scale = pl.const(1.0, pl.DT_FP16)

# Create a negative constant.
neg_offset = pl.const(-1, pl.DT_INT32)
```
