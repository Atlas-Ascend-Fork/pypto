# pypto_pro.language.distribute.Channel.submit

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

发布已暂存的WQE（doorbell）。作用范围与[Channel.is_shared](Channel-is_shared.md)一致。

## 函数原型

```python
pypto_pro.language.distribute.Channel.submit() -> None
```

## 参数说明

无。

## 返回值说明

无。

## 约束说明

- 只有在SQ和CQ还有空间时，submit=False才会延迟发布；如果新的提交会写满队列，通道会先发布此前暂存的WQE。
- submit只发布请求，不等待通信完成；需要等待完成时调用[Channel.drain](Channel-drain.md)。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_submit_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                          src_a: pl.Tensor[[64], pl.DT_UINT8],
                          src_b: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        dst = ch.remote_ptr(ws)
        # 暂存两条写请求，由一次submit统一发布。
        ch.write(dst, src_a, submit=False)
        ch.write(pl.addptr(dst, 64), src_b, submit=False)
        ch.submit()
        ch.drain()
```
