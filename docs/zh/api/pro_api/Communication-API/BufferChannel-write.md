# pypto_pro.language.distribute.BufferChannel.write

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

追加一个写操作到暂存缓冲区。

## 函数原型

```python
pypto_pro.language.distribute.BufferChannel.write(
    dst: Ptr, src: Tensor | int | float, /, *,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 远端目的地址，由基础通道的Channel.remote_ptr返回或在其基础上偏移得到，必须保持在同一URMA区域内。 |
| src | 输入 | 本地GM Tensor，必须是连续的ND行主序布局；或一个标量值。Tensor形式的数据量不能超过256 MiB，字节数必须为32的整数倍；标量形式的目标元素类型支持DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP32；目标为整型时标量必须是整型。 |
| option | 输入 | 追加WQE的BODY标志字节，编译期[UrmaOption](basic_data_structures/UrmaOption.md)常量。调用方设置的CQE位会被忽略，由[BufferChannel.submit](BufferChannel-submit.md)在最后一个WQEBB上统一设置FENCE和CQE。 |

## 返回值说明

无。

## 约束说明

- 该接口只追加，没有submit和timeout参数。缓冲区满时触发错误，不会自动提交。
- 追加操作只写UB，不访问SQ；必须调用[BufferChannel.submit](BufferChannel-submit.md)才会发布。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def buffer_channel_write_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                                src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)
        bch.write(ch.remote_ptr(ws), src)
        bch.submit()
        bch.drain()
```
