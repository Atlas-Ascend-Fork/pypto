# pypto_pro.language.distribute.barrier

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

在窗口对应通信组的所有rank之间执行软件屏障。跨rank信号交换只由每个rank的leader AIV执行（`get_block_idx() * get_subblockdim() + get_subblockid() == 0`）；sync_local控制是否在交换前后执行本rank内所有AIV的本地同步。所有rank都完成到达交换后，leader AIV上的屏障返回。

该接口只同步各rank的执行进度，不排空写引擎。如果屏障前提交了需要被对端观察到的写引擎操作，应先调用[Channel.drain](Channel-drain.md)等待写操作完成。

## 函数原型

```python
pypto_pro.language.distribute.barrier(
    win: Window,
    timeout: int | None = None,
    *,
    sync_local: bool = True,
    flush_store: bool = True,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| win | 输入 | 窗口，[pypto_pro.language.distribute.Window](Window.md)对象。屏障使用该窗口对应通信组和屏障区域。 |
| timeout | 输入 | 编译期设备cycle上限，取值范围为[0, 2^31-1]；None表示永久等待；0表示不等待；超时会触发错误。 |
| sync_local | 输入 | 是否执行一次本rank内所有AIV的本地同步，默认True。True时，调用barrier的本rank AIV在跨rank交换前后各执行一次本地同步，因此需要本rank所有AIV都调用barrier；False时不执行本地同步，只有leader AIV执行跨rank交换并等待。所有AIV都调用barrier时保持默认True；仅在单AIV或leader-only启动中使用False。 |
| flush_store | 输入 | 是否在到达交换前刷新本AIV尚未完成的MTE3写，默认True。该刷新对每个调用barrier的AIV生效，只刷新该AIV自己的MTE3写。刷新使用固定的事件ID 0，调用前后不能有其他MTE3到V的事件ID 0处于在途状态。 |

## 返回值说明

无。

## 约束说明

- 只能在Kernel的Vector执行域（[pypto_pro.language.section_vector](../SIMD-API/controlflow/section_vector.md)）内调用，不能在Cube执行域中调用。
- 这是SPMD接口：通信组内每个rank都必须启动Kernel并由leader AIV执行到该屏障。不能让某个rank跳过屏障，否则会死锁。
- sync_local=True（默认）：本rank所有AIV都要调用barrier并参与本地同步；如果有AIV未参与，本地同步无法完成。跨rank交换仍只由leader AIV执行。
- sync_local=False：不执行本rank内AIV同步，只有leader AIV等待跨rank交换完成；其他AIV调用后立即返回，不会等待该屏障。依赖该屏障的其他AIV需要在Kernel内另行同步。
- 单AIV或leader-only启动的通信模型中，由leader AIV调用barrier并设置sync_local=False，其他AIV不参与该屏障。
- flush_store=True时占用MTE3到V的事件ID 0，调用方不能在该屏障前后复用同一事件ID。
- 屏障不排空写引擎，也不保证引擎写操作对对端可见；需要时先调用[pypto_pro.language.distribute.Channel.drain](Channel-drain.md)。
- 屏障是软件屏障，开销高于核内同步接口；只在本rank内同步时使用[pypto_pro.language.system.sync_all](../SIMD-API/synchronization/sync_all.md)。

## 调用示例

### 仅leader AIV参与通信

纯Vector Kernel中一个block对应一个AIV。本示例只让block 0的leader AIV参与通信，其他AIV不调用barrier。

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit
def barrier_kernel(win: pld.Window, out: pl.Tensor[[3], pl.DT_INT64]):
    with pl.section_vector():
        if pl.get_block_idx() == 0:
            pld.barrier(win, sync_local=False)
            out[0] = 1

            pld.barrier(win, sync_local=False)
            out[1] = 2

            # 不刷新本次屏障前的MTE3写。
            pld.barrier(win, timeout=1000, sync_local=False, flush_store=False)
            out[2] = 3
```
