# pypto_pro.language.const

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

Creates a compile-time constant scalar of a specified data type. It is used in scenarios where a scalar value with an explicitly specified type is required, for example, comparing with the tile of the pypto_pro.language.DT_INT32 type or initializing an index value.

## Prototype

```python
result = pypto_pro.language.const(value, dtype, /)
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Numeric constant, which can be a Python int or float constant.|
| dtype | Input| Target data type, which can be pypto_pro.language.DT_INT8, pypto_pro.language.DT_INT16, pypto_pro.language.DT_INT32, pypto_pro.language.DT_INT64, pypto_pro.language.DT_UINT8, pypto_pro.language.DT_UINT16, pypto_pro.language.DT_UINT32, pypto_pro.language.DT_UINT64, pypto_pro.language.DT_FP16, pypto_pro.language.DT_FP32, pypto_pro.language.DT_BF16, or pypto_pro.language.DT_BOOL.|

## Constraints

None

## Return Value

Returns a scalar value, with the type specified by dtype.

## Examples

```python
import pypto_pro.language as pl
# Create a zero constant of the INT64 type.
zero_idx = pl.const(0, pl.DT_INT64)

# Create a constant of the INT32 type.
offset = pl.const(42, pl.DT_INT32)

# Create a constant of the FP16 type.
scale = pl.const(1.0, pl.DT_FP16)

# Create a negative constant.
neg_offset = pl.const(-1, pl.DT_INT32)
```
