# pypto_pro.language.distribute.BufferChannel.drain

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

等待所有已发布批次完成。

## 函数原型

```python
pypto_pro.language.distribute.BufferChannel.drain(timeout: int | None = None) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| timeout | 输入 | 编译期设备cycle上限，取值范围为[0, 2^31-1]；None表示永久等待；0表示不等待。 |

## 返回值说明

无。

## 约束说明

- 如果存在已追加但未提交的WQE，该接口触发错误。
- timeout语义与[Channel.drain](Channel-drain.md)一致；超时会触发错误。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def buffer_channel_drain_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                                src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)
        bch.write(ch.remote_ptr(ws), src)
        bch.submit()
        # 等待已发布的批次完成。
        bch.drain(timeout=1000)
```
