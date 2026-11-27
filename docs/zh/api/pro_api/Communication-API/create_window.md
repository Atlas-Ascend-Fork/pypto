# pypto_pro.language.distribute.create_window

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

在指定的HCCL通信域上创建并连接一个对称窗口。该接口为全rank集合通信接口。

窗口创建时只分配框架自带的屏障区域（一条清零的L2 Cache Line，作为region 0注册），不分配业务数据内存。业务数据通过memories参数传入已有的NPU Tensor，按列表顺序零拷贝注册到窗口，随后接口内部完成建链并返回已连接的窗口。返回的窗口可以直接用于启动Kernel，无需再调用注册或连接接口。

## 函数原型

```python
pypto_pro.language.distribute.create_window(
    group: torch.distributed.ProcessGroup,
    name: str | None = None,
    memories: Iterable[tuple[torch.Tensor, Protocol]] | None = None,
    layer: int = -1,
) -> Window
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| group | 输入 | torch.distributed进程组，后端必须为hccl。窗口的rank和nrank取自该进程组，nrank必须满足2 ≤ nrank ≤ 256。 |
| name | 输入 | 可选，窗口名称，同时作为区域标签前缀：屏障区域为"\<name>_barrier"，第i个业务区域为"\<name>_mem\<i>"；省略时使用HCCL通信域名作为前缀。同一通信域上的多个窗口必须使用不同的名称；同一窗口内区域名称不能重复，不同列表位置使用不同的_mem\<i>后缀。 |
| memories | 输入 | 可选，业务内存列表，每项为`(tensor, protocol)`二元组。tensor必须是NPU设备上的连续Tensor；protocol为[pypto_pro.language.distribute.Protocol](basic_data_structures/Protocol.md)枚举值的按位或组合，具体成员与取值参见该文档。各区域按列表顺序注册；每个区域的大小为`tensor.numel() * tensor.element_size()`，必须大于0且为32字节的整数倍。省略时不注册业务内存，窗口只包含屏障区域。 |
| layer | 输入 | HCCL网络层级。-1表示自动选择最高层，显式取值可通过[pypto_pro.language.distribute.dump_topology](dump_topology.md)查询。该参数在create_window内部建链时生效，非法层级会在本接口中报错。 |

## 返回值说明

返回已连接的[pypto_pro.language.distribute.Window](Window.md)对象，可直接作为Kernel参数启动使用。使用完毕后调用Window.close释放窗口。

## 约束说明

- 必须在所有rank上共同调用；每个rank必须传入相同顺序、相同名称（由name和列表位置决定）且大小一致的memories，否则建链失败。
- 窗口会持有memories中每个tensor的引用，直到调用[Window.close](Window.md)释放窗口。临时Tensor（如`torch.empty(...)`）无需调用方额外保活。
- 本接口返回的窗口已经完成建链，不再提供单独的注册内存或连接接口。

## 调用示例

```python
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit
def probe_kernel(win: pld.Window, out: pl.Tensor[[2], pl.DT_INT64]):
    with pl.section_vector():
        if pl.get_block_idx() == 0:
            out[0] = win.rank()
            out[1] = win.nrank()


def run(rank: int, world_size: int):
    torch.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '8361')}",
        world_size=world_size,
        rank=rank,
    )
    group = dist.distributed_c10d._get_default_group()

    ws = torch.zeros(4096, dtype=torch.uint8, device="npu")
    win = pld.create_window(
        group,
        name="probe_window",
        memories=[(ws, pld.Protocol.UB_MEM | pld.Protocol.URMA)],
    )

    out = torch.full((2,), -1, dtype=torch.int64).npu()
    probe_kernel(win, out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [rank, world_size]

    win.close()
    dist.destroy_process_group()
```
