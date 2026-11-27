# pypto_pro.language.distribute.Channel.write

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

异步写本地Tensor或标量到通道绑定的对端。

## 函数原型

```python
pypto_pro.language.distribute.Channel.write(
    dst: Ptr, src: Tensor | int | float, /, *, submit: bool = True,
    timeout: int | None = None,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 远端目的地址，由Channel.remote_ptr返回或在其基础上偏移得到，必须保持在同一URMA区域内。 |
| src | 输入 | 本地GM Tensor，或一个标量值。Tensor形式的数据量不能超过256 MiB，字节数必须为32的整数倍；Tensor必须是连续的ND行主序布局。标量形式的目标元素类型支持DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP32；目标为整型时标量必须是整型。 |
| submit | 输入 | 编译期布尔值。False只暂存WQE，True（默认）立即发布。 |
| timeout | 输入 | 编译期设备cycle上限，取值范围为[0, 2^31-1]；None表示永久等待；0表示不等待。 |
| option | 输入 | WQE控制标志，[pypto_pro.language.distribute.UrmaOption](basic_data_structures/UrmaOption.md)编译期常量，必须包含CQE。标量形式会固定设置inlineEn，option只影响顺序、fence和完成位。 |

## 返回值说明

无。

## 约束说明

- 写操作是异步的；需要确认完成时调用[Channel.drain](Channel-drain.md)。
- Channel.drain不会发布submit=False暂存的WQE；这些WQE需要调用[Channel.submit](Channel-submit.md)，或由后续提交在队列空间不足时自动发布。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_write_value_kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        # 标量形式写入远端信号字。
        ch.write(ch.remote_ptr(sig), 1, submit=True,
                 option=pld.UrmaOption.ODR_STRONG | pld.UrmaOption.FENCE | pld.UrmaOption.CQE)
        ch.drain()
```
