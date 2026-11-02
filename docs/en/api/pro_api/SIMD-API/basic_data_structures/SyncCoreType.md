# pypto_pro.language.SyncCoreType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:48:25.079Z pushedAt=2026-10-28T11:57:28.643Z nl=2 -->

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

SyncCoreType is an enumeration that specifies the core types participating in sync_all.

## Prototype

```python
PYPTO_DECLARE_ENUM(SyncCoreType,
    AIV_ONLY,  # Synchronize only the AIV cores participating in execution.
    AIC_ONLY,  # Synchronize only the AIC cores participating in execution.
    MIX        # Synchronize both the AIC cores and AIV cores participating in execution.
)
```
