# pypto_pro.language.system.dsb

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

数据同步屏障（Data Synchronization Barrier）。对本核已发出的、`scope` 范围内的访存操作进行排序：屏障前的访问全部完成后，屏障后的访问才会开始。

它只排序本核自身的访存，不同步其他核，也不等待MTE等异步pipe上尚未完成的传输——异步pipe上的posted store需要先由事件同步（sync_src/sync_dst，或pld.notify/pld.barrier内建的set_flag/wait_flag对）拉入当前pipe，再由dsb排序。典型场景：MTE3写GM完成后、写核间信号或调用sync_all之前，插入`pl.system.dsb(scope=pl.MemDsb.DDR)`保证写入先于信号到达内存系统。

## 函数原型

```python
pypto_pro.language.system.dsb(
    *,
    scope: MemDsb = pypto_pro.language.MemDsb.ALL,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| scope | 输入 | 可选，[pypto_pro.language.MemDsb](../basic_data_structures/MemDsb.md)枚举值，指定屏障排序的访存范围。默认为pypto_pro.language.MemDsb.ALL（排序全部访存）；仅需要排序GM访问时使用DDR以降低开销。 |

## 约束说明

- dsb仅排序本核访存，不提供核间同步；核间同步请使用sync_all或set_cross_core/wait_cross_core。dsb负责的是屏障两侧本核访存的先后顺序。
- 异步pipe（MTE*）上的posted store不会被dsb直接等待。需要“MTE3写GM先于后续信号可见”时，先做MTE3事件同步（例如sync_src/sync_dst，或pld.barrier(flush_store=True)内建的对），再调用dsb。
- dsb在S pipe上发射。与bar_*系列不同，dsb不是pipe屏障，不会等待某条pipe队列排空。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit
def dsb_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x0200)
    pl.load(tile_a, a, [0, 0])
    pl.add(tile_out, tile_a, tile_a)
    pl.store(out, tile_out, [0, 0])

    # MTE3的store完成后，排序GM访问（本核），后续（如跨核信号写入）可见此前写入。
    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=0)
    pl.system.dsb(scope=pl.MemDsb.DDR)
```

该示例在store之后用MTE3→S事件对排空MTE3队列，随后dsb(DDR)保证本核后续访存（示例中未画出的信号写入，参见pypto_distribute.h的notify实现）排在GM写入之后。
