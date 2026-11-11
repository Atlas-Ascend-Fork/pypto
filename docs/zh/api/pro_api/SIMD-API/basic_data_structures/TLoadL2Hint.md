# pypto_pro.language.TLoadL2Hint

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

GM数据读取的L2 Cache策略枚举，用于load和load_tile接口。该参数仅影响缓存策略，不改变搬运结果。

## 原型定义

```python
PYPTO_DECLARE_ENUM(TLoadL2Hint,
    NormalFirstVictim,  # 启用L2 Cache，并且将分配的Cache Line标记为高替换优先级。
    NormalLastVictim,   # 启用L2 Cache，并且将分配的Cache Line标记为低替换优先级。
    NormalPersistent,   # 启用L2 Cache；被标记为驻留模式的Cache Line只能被其他同样标记为驻留模式的Cache Line替换。
    NotAllocKeep,       # 不启用L2 Cache，每次直接从GM读取，并且保持已有Cache Line的状态不变。
    NotAllocClean,      # 不启用L2 Cache；若L2 Cache中已有同地址缓存，则保留并标记为Clean，移出时直接丢弃。
    NotAllocDrop        # 不启用L2 Cache；若L2 Cache中已有同地址缓存，则直接丢弃。
)
```
