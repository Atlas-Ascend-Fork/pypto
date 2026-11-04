# pypto_pro.language.TStoreL2Hint

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

GM数据写入的L2 Cache策略枚举，用于store和store_tile接口。该参数仅影响缓存策略，不改变搬运结果。

## 原型定义

```python
PYPTO_DECLARE_ENUM(TStoreL2Hint,
    NormalFirstVictim,  # 默认策略，正常分配L2 Cache，优先淘汰。
    NormalLastVictim,   # 正常分配L2 Cache，降低被淘汰的优先级。
    NormalPersistent,   # 正常分配L2 Cache，并使用持久化策略。
    NotAllocClean       # 不分配新的L2 Cache行，并使用Clean策略。
)
```
