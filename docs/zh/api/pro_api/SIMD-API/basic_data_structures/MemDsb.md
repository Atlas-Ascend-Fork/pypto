# pypto_pro.language.MemDsb

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

数据同步屏障（dsb）作用域枚举，用于指定 `pl.system.dsb` 排序的访存范围。枚举值对应工具链 `mem_dsb_t`（cce_aicore_intrinsics.h）中的DSB_ALL/DSB_DDR/DSB_UB，代码生成时加上DSB_前缀映射为对应硬件指令。

## 原型定义

```python
PYPTO_DECLARE_ENUM(MemDsb,
    ALL,  # 排序所有访存指令（默认）。
    DDR,  # 仅排序访问DDR的访存指令。
    UB,   # 仅排序访问UB的访存指令。
)
```

显式指定枚举值时，请与实际需要排序的访存路径匹配；仅需要保证GM写入可见时使用DDR，具体约束请参见[dsb](../synchronization/dsb.md)。
