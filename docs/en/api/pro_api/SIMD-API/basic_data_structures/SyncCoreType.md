# pypto_pro.language.SyncCoreType

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

SyncCoreType enumerates the core types that participate in sync_all.

## Prototype

```python
PYPTO_DECLARE_ENUM(SyncCoreType,
    AIV_ONLY,  # Only the AIV cores involved in execution are synchronized.
    AIC_ONLY,  # Only the AIC cores involved in execution are synchronized.
    MIX        # Both the AIC and AIV cores involved in execution are synchronized.
)
```
