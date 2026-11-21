# pypto_pro.language.CrossCoreSyncMode

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

Enumerates the inter-core synchronization modes. It is used to specify the cores involved in synchronization of [set_cross_core](../synchronization/set_cross_core.md) and [wait_cross_core](../synchronization/wait_cross_core.md) as well as the pairing mode of SET/WAIT signals.

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

| Value| Description|
|:-------|:-----|
| INTER_BLOCK | The mode value is 0. It synchronizes all cores of the same type between multiple AI Cores. In the AI Cube scenario, all AI Cubes started by the current kernel are synchronized. In the AI Vector scenario, all AI Vectors started by the current kernel are synchronized. In this mode, AI Cubes and AI Vectors do not synchronize with each other.|
| INTER_SUBBLOCK | The mode value is 1. It synchronizes AIV0 and AIV1 within the same AI Core. Different AI Cores do not affect each other.|
| INTRA_BLOCK | The mode value is 2. It synchronizes the AI Cube with all AI Vectors within the same AI Core. In the direction from AI Vector to AI Cube, AIV0 and AIV1 send signals separately, and the AI Cube waits for the two signals. In the direction from AI Cube to AI Vector, the AI Cube sends signals, and AIV0 and AIV1 wait for the signals separately. This is the default synchronization mode of set_cross_core and wait_cross_core.|
| UNICAST_BLOCK | The mode value is 3. It synchronizes the AI Cube with a single AI Vector within the same AI Core. On the AI Cube side, event IDs 0–15 correspond to AIV0, and event IDs 16–31 correspond to AIV1. On the AI Vector side, event IDs range from 0 to 15.|

For details about the event ID range, pipeline restrictions, and SET/WAIT pairing requirements in each mode, see [set_cross_core](../synchronization/set_cross_core.md) and [wait_cross_core](../synchronization/wait_cross_core.md).
