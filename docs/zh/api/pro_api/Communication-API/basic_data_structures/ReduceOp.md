# pypto_pro.language.distribute.ReduceOp

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

[Channel.write_reduce](../Channel-write_reduce.md)和[BufferChannel.write_reduce](../BufferChannel-write_reduce.md)的逐元素归约模式。枚举值为引擎的线缆编码，声明顺序与硬件枚举一致。

## 原型定义

```python
class ReduceOp(enum.IntEnum):
    MAX = ...  # 逐元素取最大值
    MIN = ...  # 逐元素取最小值
    SUM = ...  # 逐元素求和
```

## 约束说明

- 归约源必须是支持的数据类型：DT_INT8、DT_INT16、DT_INT32、DT_UINT32、DT_FP16、DT_FP32、DT_BF16。
- 归约源的数据类型必须与远端目标指针的元素类型一致。
- op必须显式指定，没有默认值；不支持省略。
- 多个rank同时归约到同一目标地址时，硬件会串行执行，但结果与其他引擎操作之间没有顺序保证；需要先调用对应通道的drain方法。
- 单次归约的数据量不能超过256 MiB，超过时需要调用方拆分。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def reduce_sum_kernel(win: pld.Window, ws: pl.Tensor[[256], pl.DT_UINT8],
                      src: pl.Tensor[[16], pl.DT_INT32]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()
        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        dst = pl.make_ptr(ch.remote_ptr(ws), pl.DT_INT32)
        ch.write_reduce(dst, src, op=pld.ReduceOp.SUM)
        ch.drain()
```
