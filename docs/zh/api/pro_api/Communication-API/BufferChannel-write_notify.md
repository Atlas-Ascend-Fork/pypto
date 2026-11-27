# pypto_pro.language.distribute.BufferChannel.write_notify

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

追加一个带通知的写操作到暂存缓冲区，占用两个WQEBB。

## 函数原型

```python
pypto_pro.language.distribute.BufferChannel.write_notify(
    dst: Ptr, src: Tensor, signal: Ptr, val: int, /, *,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 远端目的地址，由基础通道的Channel.remote_ptr返回或在其基础上偏移得到，必须保持在同一URMA区域内。 |
| src | 输入 | 本地GM Tensor，必须是连续的ND行主序布局。数据量不能超过256 MiB，字节数必须为32的整数倍。 |
| signal | 输出 | 对端64位信号字的URMA地址，必须通过Channel.remote_ptr解析DT_INT64或DT_UINT64 Tensor得到，并与dst保持在同一URMA区域内。 |
| val | 输入 | 硬件在数据写完成后写入信号字的通知值，整型标量或整型标量表达式。 |
| option | 输入 | 追加WQE的BODY标志字节，编译期[UrmaOption](basic_data_structures/UrmaOption.md)常量，CQE位会被忽略。 |

## 返回值说明

无。

## 约束说明

- 通知在批次发布后按顺序触发；数据先于信号可见。
- 该接口只追加，不发布；必须调用[BufferChannel.submit](BufferChannel-submit.md)才会发布。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def buffer_channel_write_notify_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                                       sig: pl.Tensor[[16], pl.DT_INT64],
                                       src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)
        bch.write_notify(ch.remote_ptr(ws), src,
                         pl.addptr(ch.remote_ptr(sig), me), 1)
        bch.submit()
        bch.drain()
```
