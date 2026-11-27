# pypto_pro.language.distribute.BufferChannel.read

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

追加一个读操作到暂存缓冲区。

## 函数原型

```python
pypto_pro.language.distribute.BufferChannel.read(
    dst: Tensor, src: Ptr, /, *,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 本地GM Tensor，接收读取的数据。数据量不能超过256 MiB，字节数必须为32的整数倍；Tensor必须是连续的ND行主序布局。读取的数据由引擎写入，只能在[BufferChannel.drain](BufferChannel-drain.md)之后读取。 |
| src | 输入 | 远端源地址，由基础通道的Channel.remote_ptr返回或在其基础上偏移得到，必须保持在同一URMA区域内。 |
| option | 输入 | 追加WQE的BODY标志字节，编译期[UrmaOption](basic_data_structures/UrmaOption.md)常量，CQE位会被忽略。 |

## 返回值说明

无。

## 约束说明

- 读取结果为异步写入，需在[BufferChannel.drain](BufferChannel-drain.md)之后使用。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def buffer_channel_read_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                               out: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)
        bch.read(out, ch.remote_ptr(ws))
        bch.submit()
        bch.drain()
```
