# pypto_pro.language.distribute.BufferChannel.write_reduce

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

追加一个逐元素归约操作到暂存缓冲区。

## 函数原型

```python
pypto_pro.language.distribute.BufferChannel.write_reduce(
    dst: Ptr, src: Tensor, /, *, op: ReduceOp,
    option: UrmaOption = UrmaOption.DEFAULT) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| dst | 输出 | 远端目的地址，由Channel.remote_ptr解析得到，必须保持在同一URMA区域内；元素类型必须与src一致。 |
| src | 输入 | 本地GM Tensor，必须是连续的ND行主序布局。支持DT_INT8、DT_INT16、DT_INT32、DT_UINT32、DT_FP16、DT_FP32、DT_BF16。单次数据量不能超过256 MiB，字节数必须为32的整数倍。 |
| op | 输入 | 归约模式，[pypto_pro.language.distribute.ReduceOp](basic_data_structures/ReduceOp.md)枚举值，必填。 |
| option | 输入 | 追加WQE的BODY标志字节，编译期[UrmaOption](basic_data_structures/UrmaOption.md)常量，CQE位会被忽略。 |

## 返回值说明

无。

## 约束说明

- 多个rank同时归约到同一目标地址时，硬件会串行执行；归约结果与其他引擎操作之间没有顺序保证，需要先调用[BufferChannel.drain](BufferChannel-drain.md)。
- 该接口只追加，不发布；必须调用[BufferChannel.submit](BufferChannel-submit.md)才会发布。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def buffer_channel_write_reduce_kernel(win: pld.Window, ws: pl.Tensor[[256], pl.DT_UINT8],
                                       src: pl.Tensor[[16], pl.DT_INT32]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        bch = pld.buffer_channel(ch, 0x4000, 4096)
        dst = pl.make_ptr(ch.remote_ptr(ws), pl.DT_INT32)
        bch.write_reduce(dst, src, op=pld.ReduceOp.SUM)
        bch.submit()
        bch.drain()
```
