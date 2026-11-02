# SaturationFlagMode

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:30:49.306Z pushedAt=2026-10-28T11:57:28.887Z nl=2 -->

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

SaturationFlagMode defines the saturation mode categories for the [pypto_pro.language.set_saturation_flag](../special_reg_access/set_saturation_flag.md) and [pypto_pro.language.get_saturation_flag](../special_reg_access/get_saturation_flag.md) operations, used to control the saturation behavior of different computation scenarios in the CTRL register.

Each mode corresponds to a specific bit in the CTRL register:

| Enum Value | CTRL Register Bit | Control Range | Polarity |
|---|---|---|---|
| FLOAT | bit 48 | Floating-point computation and floating-point precision conversion | Inverted (bit=0 enables saturation) |
| FLOAT8 | bit 50 | Float8 computation | Inverted (bit=0 enables saturation) |
| INT | bit 53 | Integer computation | Normal (bit=1 enables saturation) |
| CAST | bit 59 | Precision conversion from floating-point to integer or integer to integer | Inverted (bit=0 enables saturation) |

## Prototype

```python
class SaturationFlagMode(enum.Enum):
    FLOAT = ...    # Floating-point computation and floating-point precision conversion (CTRL bit 48).
    FLOAT8 = ...   # Float8 computation (CTRL bit 50).
    INT = ...      # Integer computation (CTRL bit 53).
    CAST = ...     # Float-to-integer/integer-to-integer precision conversion (CTRL bit 59).
```

## Constraints

- When used together with the [SaturateMode](SaturateMode.md) parameter of [vf.astype](../type_conversion/astype.md), the single-instruction mode applies when CTRL[60]=0 (controlled by the saturate parameter), and the global mode applies when CTRL[60]=1 (the CTRL bit controlled by this enum takes effect).

## Examples

```python
import pypto_pro.language as pl

@pl.jit()
def kernel():
    # Set the saturation mode of the CAST category to global enable before VF computation.
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    # ... Perform VF computation ...
    # Restore.
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=False)
```
