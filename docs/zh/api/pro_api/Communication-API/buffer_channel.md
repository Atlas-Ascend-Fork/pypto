# pypto_pro.language.distribute.buffer_channel

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

在HCCL通信域的窗口上，将一个调用方预留的UB暂存缓冲区绑定到已打开的通道，形成批量通道。批量通道把多个操作追加到调用方预留的UB暂存缓冲区，最后通过一次BufferChannel.submit批量发布。追加操作只执行UB写，不访问SQ；发布时执行一次批量DMA、一次头部推进和一次doorbell，整个批次只产生一个完成记录。

批量通道复用基础通道的Jetty、对端和头部信息，因此二者必须配套使用：通过[pypto_pro.language.distribute.channel](channel.md)打开基础通道，再通过本接口绑定暂存缓冲区。

## 函数原型

```python
pypto_pro.language.distribute.buffer_channel(
    ch: Channel,
    buf_addr: int,
    buf_bytes: int,
    /,
) -> BufferChannel
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| ch | 输入 | 已打开的通道，[pypto_pro.language.distribute.channel](channel.md)返回的句柄。 |
| buf_addr | 输入 | 暂存缓冲区的UB起始地址，编译期整数，必须按64字节对齐，且不能与Tile或其他通道的暂存区重叠。 |
| buf_bytes | 输入 | 暂存缓冲区大小，编译期整数，至少为128字节且为64字节的整数倍，且buf_bytes / 64不能超过Jetty的SQ深度。带通知的写操作和原子操作的WQE各占两个64字节槽位。 |

## 返回值说明

返回pypto_pro.language.distribute.BufferChannel解析期句柄。其write、read、write_reduce、write_notify和原子方法只追加WQE到暂存缓冲区，由[BufferChannel.submit](BufferChannel-submit.md)统一发布。句柄字段如下：

| 字段 | 说明 |
|---|---|
| ch | 基础通道，批量通道复用其Jetty、对端和头部信息。 |
| buf_addr | 暂存缓冲区UB起始地址。 |
| buf_bytes | 暂存缓冲区大小，单位为字节。 |
| token | 解析器生成的批量通道标识。 |

## 约束说明

- BufferChannel只依赖标量操作，只能在Kernel的Vector执行域中使用，不能作为普通Python对象构造。
- 批量通道必须在简单赋值中创建，不能放在控制流内部。
- ch必须是[pypto_pro.language.distribute.channel](channel.md)返回的通道句柄，不能是其他表达式。
- buf_addr和buf_bytes必须是编译期常量；缓冲区不能与Tile或其他通道的暂存区重叠，框架会按`buf_addr + buf_bytes`预留UB空间；缓冲区的WQEBB数（buf_bytes / 64）不能超过Jetty的SQ深度。
- 缓冲区满时追加WQE会触发错误，不会自动提交；调用方需要及时调用[BufferChannel.submit](BufferChannel-submit.md)。
- 批量未提交时不能调用基础通道的发起类方法（write/read/write_reduce/write_notify/原子操作），否则会因头部状态不一致而触发错误；ch.submit()只响铃，不会改变头部状态。BufferChannel.submit之后可以正常使用基础通道。
- [BufferChannel.drain](BufferChannel-drain.md)在存在未提交的WQE时触发错误。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def batch_write_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                       src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)

        dst = ch.remote_ptr(ws)
        for i in pl.range(16):
            # 追加到暂存缓冲区，不立即提交。
            bch.write(pl.addptr(dst, i * 64), src)

        # 发布整批WQE并等待完成。
        bch.submit()
        bch.drain()
```
