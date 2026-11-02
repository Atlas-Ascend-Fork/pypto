# pypto_pro.language.DcciDst

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:45:10.772Z pushedAt=2026-10-28T11:57:28.614Z nl=2 -->

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

Enumeration of DCCI consistency targets, used to specify the storage area or hardware access path that must be kept consistent with the Data Cache.

## Prototype

```python
PYPTO_DECLARE_ENUM(DcciDst,
    AUTO,             # Automatically select based on the target object: CACHELINE_OUT for GM Tensor and CACHELINE_UB for UB Tile.
    CACHELINE_OUT,    # Ensure consistency between the Data Cache and GM. Applicable to GM Tensor.
    CACHELINE_UB,     # Ensure consistency between the Data Cache and UB. Applicable to UB Tile.
    CACHELINE_ALL,    # Same effect as CACHELINE_OUT. Applicable to GM Tensor.
    CACHELINE_ATOMIC  # Ensure consistency between the Data Cache and GM on the hardware atomic cache path.
)
```

When explicitly specifying an enum value, it must match the storage area and hardware access path of the target object. For specific constraints, see [dcci](../cache_control/dcci.md).
