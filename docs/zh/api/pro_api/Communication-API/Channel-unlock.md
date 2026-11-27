# pypto_pro.language.distribute.Channel.unlock

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

释放通道的跨AIV锁。

释放前会把当前AIV缓存的生产者状态发布到共享计数器。共享窗口下为空操作。

## 函数原型

```python
pypto_pro.language.distribute.Channel.unlock() -> None
```

## 参数说明

无。

## 返回值说明

无。

## 约束说明

- 只允许与[Channel.lock](Channel-lock.md)成对使用。
- 释放前应确认由该锁保护的post、submit、drain序列已完成。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_unlock_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                          src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        ch.lock()
        ch.write(ch.remote_ptr(ws), src, submit=False)
        ch.submit()
        ch.drain()
        ch.unlock()
```
