# pypto_pro.language.distribute.Channel.read

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

异步从通道绑定的对端读取数据到本地Tensor。

## 函数原型

```python
pypto_pro.language.distribute.Channel.read(
    dst: Tensor, src: Ptr, /, *, submit: bool = True,
    timeout: int | None = None,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 本地GM Tensor，接收读取的数据。数据量不能超过256 MiB，字节数必须为32的整数倍；Tensor必须是连续的ND行主序布局。引擎写入dst，内容只能在[Channel.drain](Channel-drain.md)之后读取。 |
| src | 输入 | 远端源地址，由Channel.remote_ptr返回或在其基础上偏移得到，必须保持在同一URMA区域内。 |
| submit | 输入 | 编译期布尔值。False只暂存WQE，True（默认）立即发布。 |
| timeout | 输入 | 编译期设备cycle上限，取值范围为[0, 2^31-1]；None表示永久等待；0表示不等待。 |
| option | 输入 | WQE控制标志，必须包含CQE。 |

## 返回值说明

无。

## 约束说明

- 读取结果为异步写入，需在[Channel.drain](Channel-drain.md)之后使用。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_read_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                        out: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        # 从对端读取64字节到本地Tensor，等待通信完成后使用。
        ch.read(out, ch.remote_ptr(ws))
        ch.drain()
```
