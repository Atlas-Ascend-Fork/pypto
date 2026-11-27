# pypto_pro.language.distribute.dump_topology

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

查询并返回指定HCCL通信域的物理拓扑报告。该接口为主机侧操作，不是集合通信接口。

输出内容包括每个网络层级的互连拓扑（如服务器内的1DMesh、服务器间的CLOS）、当前rank在该层可到达的rank，以及到每个对端的物理链路信息（传输协议、跳数、本端和对端地址）。

## 函数原型

```python
pypto_pro.language.distribute.dump_topology(
    group: torch.distributed.ProcessGroup,
    layer: int = -1,
) -> str
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| group | 输入 | torch.distributed进程组，后端必须为hccl。 |
| layer | 输入 | 只输出指定的网络层级。-1输出全部层级；0通常为服务器内互连，1通常为服务器间网络。也可以使用[pypto_pro.language.distribute.create_window](create_window.md)的layer参数指定窗口使用的层级。 |

## 返回值说明

返回拓扑报告字符串。报告包含以下字段：

| 字段 | 说明 |
|---|---|
| Local Rank | 当前rank及其所在通信组的rank总数。 |
| Network Layer | 网络层级编号。 |
| Topology | 该层的互连拓扑类型。 |
| Reachable | 当前rank在该层可到达的rank数量。 |
| Peer、Transport、Hops | 对端rank、传输协议和跳数。 |
| LocalDevice、LocalAddress、RemoteDevice、RemoteAddress | 本端和对端的设备及地址信息。 |

## 约束说明

- group必须是已初始化的HCCL进程组，且对应的HCCL comm可用。
- 指定的layer必须存在于该通信域的rank graph中，否则报错。
- 该接口只读取拓扑信息，不创建窗口，也不改变通信状态。

## 调用示例

```python
import torch.distributed as dist
import pypto_pro.language.distribute as pld


def dump(group: dist.ProcessGroup) -> str:
    # 输出全部网络层级的拓扑。
    report = pld.dump_topology(group)
    print(report)
```
