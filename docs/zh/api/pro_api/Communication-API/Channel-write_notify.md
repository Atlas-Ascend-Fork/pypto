# pypto_pro.language.distribute.Channel.write_notify

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

写操作完成时由硬件向对端信号字写通知值的融合接口。

不建议将其作为批量或多条消息场景的默认通知方式：它只保证本条消息和通知值同时可见，不能建立与前序其它写请求的顺序。多条数据需要通知同一消费者时，应使用前面的写请求配合最后一条ODR_STRONG | FENCE | CQE写发布，批量场景优先使用[BufferChannel](buffer_channel.md)。

## 函数原型

```python
pypto_pro.language.distribute.Channel.write_notify(
    dst: Ptr, src: Tensor, signal: Ptr, val: int, /, *,
    submit: bool = True, timeout: int | None = None,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 远端目的地址，由Channel.remote_ptr返回或在其基础上偏移得到，必须保持在同一URMA区域内。 |
| src | 输入 | 本地GM Tensor，必须是连续的ND行主序布局。数据量不能超过256 MiB，字节数必须为32的整数倍；该接口没有标量形式。 |
| signal | 输出 | 对端64位信号字的URMA地址。必须通过Channel.remote_ptr解析一个DT_INT64或DT_UINT64 Tensor得到，并与dst保持在同一URMA区域内；可以用[pypto_pro.language.addptr](../SIMD-API/resource_management/addptr.md)偏移到具体槽位。32位指针会被拒绝。 |
| val | 输入 | 硬件在数据写完成后写入信号字的通知值，整型标量或整型标量表达式。 |
| submit | 输入 | 编译期布尔值。False只暂存WQE，True（默认）立即发布。 |
| timeout | 输入 | 编译期设备cycle上限，取值范围为[0, 2^31-1]；None表示永久等待；0表示不等待。 |
| option | 输入 | WQE控制标志，必须包含CQE。 |

## 返回值说明

无。

## 约束说明

- 硬件在数据落地后才写信号字，因此本条数据先于信号可见。如果改用独立信号写，信号写应带ODR_STRONG | FENCE | CQE，以建立与前序数据写的顺序。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_write_notify_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                                sig: pl.Tensor[[16], pl.DT_INT64],
                                src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        ch.write_notify(ch.remote_ptr(ws), src,
                        pl.addptr(ch.remote_ptr(sig), me), 1)
        ch.drain()
```
