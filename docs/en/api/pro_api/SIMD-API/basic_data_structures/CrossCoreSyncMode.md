# pypto_pro.language.CrossCoreSyncMode

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:43:21.150Z pushedAt=2026-10-28T11:57:28.602Z nl=2 -->

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

Enumeration of inter-core synchronization modes, used to specify the cores participating in synchronization in [set_cross_core](../synchronization/set_cross_core.md) and [wait_cross_core](../synchronization/wait_cross_core.md), as well as the pairing method of SET/WAIT signals.

## Prototype

```python
PYPTO_DECLARE_ENUM(CrossCoreSyncMode,
    INTER_BLOCK,
    INTER_SUBBLOCK,
    INTRA_BLOCK,
    UNICAST_BLOCK
)
```

## Parameters

| Parameter Value | Description |
|:-------|:-----|
| INTER_BLOCK | The mode value is 0. Full-core synchronization among cores of the same type across multiple AI Cores. In the AIC scenario, all AICs launched by this Kernel are synchronized; in the AIV scenario, all AIVs launched by this Kernel are synchronized. AIC and AIV do not synchronize with each other in this mode. |
| INTER_SUBBLOCK | The mode value is 1. AIV0 and AIV1 within the same AI Core are synchronized, and different AI Cores do not affect each other. |
| INTRA_BLOCK | The mode value is 2. The AIC and all AIVs within the same AI Core are synchronized. In the AIV-to-AIC direction, AIV0 and AIV1 must send signals separately, and the AIC waits for both signals. In the AIC-to-AIV direction, the AIC sends a signal, and AIV0 and AIV1 wait separately. This value is the default synchronization mode of set_cross_core and wait_cross_core. |
| UNICAST_BLOCK | The mode value is 3. The AIC and a single AIV within the same AI Core are synchronized. On the AIC side, event IDs 0 to 15 correspond to AIV0, and 16 to 31 correspond to AIV1. On the AIV side, event IDs range from 0 to 15. |

For the specific event ID ranges, pipeline restrictions, and the SET/WAIT pairing requirements of each mode, see [set_cross_core](../synchronization/set_cross_core.md) and [wait_cross_core](../synchronization/wait_cross_core.md).
