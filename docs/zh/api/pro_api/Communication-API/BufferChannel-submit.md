# pypto_pro.language.distribute.BufferChannel.submit

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

发布整个暂存批次。

## 函数原型

```python
pypto_pro.language.distribute.BufferChannel.submit(timeout: int | None = None) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| timeout | 输入 | 编译期设备cycle上限，取值范围为[0, 2^31-1]；None表示永久等待；0表示不等待。该上限约束等待SQ/CQ空间的时间。 |

## 返回值说明

无。

## 约束说明

- 发布过程执行一次批量DMA、一次头部推进和一次doorbell，最后一个WQEBB携带FENCE和CQE，整个批次只产生一个完成记录。
- doorbell总会触发，因此此前通过基础通道submit=False提交的操作也会被发布。
- submit只发布请求，不等待通信完成；需要等待完成时调用[BufferChannel.drain](BufferChannel-drain.md)。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def buffer_channel_submit_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                                 src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)
        dst = ch.remote_ptr(ws)
        bch.write(dst, src)
        bch.write(pl.addptr(dst, 64), src)
        # 一次submit发布整个批次。
        bch.submit(timeout=1000)
        bch.drain()
```
