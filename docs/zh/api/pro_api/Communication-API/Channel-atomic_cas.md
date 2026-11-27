# pypto_pro.language.distribute.Channel.atomic_cas

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

对远端一个对齐的整数执行compare-and-swap。

## 函数原型

```python
pypto_pro.language.distribute.Channel.atomic_cas(
    dst: Ptr, fetch: Tensor, expected: int, desired: int, /, *,
    submit: bool = True, timeout: int | None = None,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 远端目标字地址，由Channel.remote_ptr返回或在其基础上偏移得到，必须保持在同一URMA区域内。要求与[atomic_fetch_add](Channel-atomic_fetch_add.md)相同：DT_INT32、DT_UINT32、DT_INT64或DT_UINT64，自然对齐。 |
| fetch | 输出 | 本地单元素GM Tensor，元素类型与dst一致，用于接收远端旧值。无论比较是否成功都会写入远端旧值，当fetch == expected时表示交换成功；旧值只能在[Channel.drain](Channel-drain.md)之后读取。 |
| expected | 输入 | 期望的远端旧值，整型标量或整型标量表达式。 |
| desired | 输入 | 比较成功时写入的新值，整型标量或整型标量表达式。 |
| submit | 输入 | 编译期布尔值。False只暂存WQE，True（默认）立即发布。 |
| timeout | 输入 | 编译期设备cycle上限，取值范围为[0, 2^31-1]；None表示永久等待；0表示不等待。 |
| option | 输入 | WQE控制标志，必须包含CQE。 |

## 返回值说明

无。

## 约束说明

- 原子操作是异步的；需要确认完成时调用[Channel.drain](Channel-drain.md)。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_atomic_cas_kernel(win: pld.Window, sig: pl.Tensor[[16], pl.DT_INT64],
                              fetch: pl.Tensor[[1], pl.DT_INT64]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        ch.atomic_cas(ch.remote_ptr(sig), fetch, 1, 2)
        ch.drain()
        # drain之后通过fetch确认是否交换成功。
```
