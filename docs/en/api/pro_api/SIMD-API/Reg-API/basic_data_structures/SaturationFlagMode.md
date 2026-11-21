# SaturationFlagMode

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

SaturationFlagMode defines the saturation mode categories for [pypto_pro.language.set_saturation_flag](../special_reg_access/set_saturation_flag.md) and [pypto_pro.language.get_saturation_flag](../special_reg_access/get_saturation_flag.md) operations. It controls the saturation behavior of different computation scenarios in the CTRL register.

Each mode corresponds to a specific bit in the CTRL register.

| Enumerated Value| CTRL Register Bit| Control Scope| Polarity|
|---|---|---|---|
| FLOAT | bit 48 | Floating-point computation and floating-point precision conversion| Inverted (saturation enabled when bit = 0)|
| FLOAT8 | bit 50 | FP8 computation| Inverted (saturation enabled when bit = 0)|
| INT | bit 53 | Integer computation| Normal (saturation enabled when bit = 1)|
| CAST | bit 59 | Precision conversion from floating-point to integer or from integer to integer| Inverted (saturation enabled when bit = 0)|

## Prototype

```python
class SaturationFlagMode(enum.Enum):
    FLOAT = ...    # Floating-point computation and floating-point precision conversion (CTRL bit 48)
    FLOAT8 = ...   # FP8 computation (CTRL bit 50)
    INT = ...      # Integer computation (CTRL bit 53)
    CAST = ...     # Precision conversion from floating-point to integer or from integer to integer (CTRL bit 59)
```

## Constraints

- When SaturationFlagMode is used with the [SaturateMode](SaturateMode.md) parameter of [vf.astype](../type_conversion/astype.md), if CTRL[60] = 0, the single-instruction mode is used (controlled by the saturate parameter). If CTRL[60] = 1, the global mode is used (the CTRL bits controlled by this enumeration take effect).

## Examples

```python
import pypto_pro.language as pl

@pl.jit()
def kernel():
    # Enable global saturation mode for the CAST category before VF computation.
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    # ... Perform VF computation ...
    # Restore.
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=False)
```
