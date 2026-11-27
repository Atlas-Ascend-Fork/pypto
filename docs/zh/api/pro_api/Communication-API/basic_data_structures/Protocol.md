# pypto_pro.language.distribute.Protocol

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

内存区域注册的引擎平面选择，用于[pypto_pro.language.distribute.create_window](../create_window.md)的memories参数。

Protocol是enum.IntFlag，多个平面可以通过按位或组合使用。

## 原型定义

```python
class Protocol(enum.IntFlag):
    UB_MEM = ...  # UB直连平面，注册后可通过remote_ptr/remote_tensor直接访问对端内存
    URMA = ...  # URMA写引擎平面，注册后可作为Channel的通道区域，提交异步写、读、归约和原子操作
```

## 参数说明

无。

## 约束说明

- memories中每个区域的protocol不能为0，且只能包含UB_MEM和URMA两个位。
- 只使用pypto_pro.language.distribute.remote_ptr或pypto_pro.language.distribute.remote_tensor的区域在访问对端时必须包含Protocol.UB_MEM，否则运行时触发错误。
- 只使用pypto_pro.language.distribute.channel写引擎的区域必须包含Protocol.URMA；窗口至少要有一个URMA区域才能打开通道。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language.distribute as pld


def create_window_for(group, tensor):
    # 同一区域同时注册UB_MEM和URMA两个平面。
    return pld.create_window(
        group,
        memories=[(tensor, pld.Protocol.UB_MEM | pld.Protocol.URMA)],
    )
```
