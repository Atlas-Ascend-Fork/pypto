# pypto_pro.language.distribute.Channel.lock

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

获取通道的跨AIV锁。

共享窗口使用当前AIV自己的Jetty，该接口立即返回。非共享窗口中，同一对端的Jetty由多个AIV共享，该接口会自旋等待通道锁字，并在加锁后从共享计数器重新加载生产者头部。

## 函数原型

```python
pypto_pro.language.distribute.Channel.lock() -> None
```

## 参数说明

无。

## 返回值说明

无。

## 约束说明

- 锁的范围应覆盖完整的post、submit、drain序列，以及随该序列发布的暂存追加操作。
- 只允许与[Channel.unlock](Channel-unlock.md)成对使用；共享窗口下为空操作。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_lock_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
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
