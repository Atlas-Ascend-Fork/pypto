# pypto_pro.language.distribute.BufferChannel.atomic_fetch_add

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

追加一个fetch-and-add操作到暂存缓冲区，占用两个WQEBB。

## 函数原型

```python
pypto_pro.language.distribute.BufferChannel.atomic_fetch_add(
    dst: Ptr, fetch: Tensor, val: int, /, *,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 远端目标字地址，由Channel.remote_ptr解析得到，必须保持在同一URMA区域内。目标必须是DT_INT32、DT_UINT32、DT_INT64或DT_UINT64，并按元素大小自然对齐。 |
| fetch | 输出 | 本地单元素GM Tensor，元素类型与dst一致，用于接收远端旧值。旧值只能在[BufferChannel.drain](BufferChannel-drain.md)之后读取。 |
| val | 输入 | 加数，整型标量或整型标量表达式。 |
| option | 输入 | 追加WQE的BODY标志字节，编译期[UrmaOption](basic_data_structures/UrmaOption.md)常量，CQE位会被忽略。 |

## 返回值说明

无。

## 约束说明

- 该接口只追加，不发布；必须调用[BufferChannel.submit](BufferChannel-submit.md)才会发布。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def buffer_channel_atomic_fetch_add_kernel(win: pld.Window,
                                           sig: pl.Tensor[[16], pl.DT_INT64],
                                           fetch: pl.Tensor[[1], pl.DT_INT64]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)
        bch.atomic_fetch_add(ch.remote_ptr(sig), fetch, 1)
        bch.submit()
        bch.drain()
```
