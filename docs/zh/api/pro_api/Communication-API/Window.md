# pypto_pro.language.distribute.Window

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

对称窗口是HCCL通信域内各rank共享的通信上下文，Window是它的主机侧句柄，由[pypto_pro.language.distribute.create_window](create_window.md)创建。窗口在创建时完成业务内存注册和建链，返回后即可直接用于启动Kernel。

Window提供以下能力：

- 查询当前rank、rank总数和窗口拓扑类型。
- 查询设备侧窗口描述符地址。
- 作为Kernel参数传入设备侧，供[barrier](barrier.md)、[channel](channel.md)、[remote_ptr](remote_ptr.md)等接口使用。
- 释放窗口及其持有的业务Tensor引用。

该类型不能直接实例化，必须通过[pypto_pro.language.distribute.create_window](create_window.md)获取。

## 函数原型

```python
pypto_pro.language.distribute.Window.rank() -> int
pypto_pro.language.distribute.Window.nrank() -> int
pypto_pro.language.distribute.Window.data_ptr() -> int
pypto_pro.language.distribute.Window.is_shared() -> bool
pypto_pro.language.distribute.Window.close() -> None
```

## 参数说明

构造参数由[pypto_pro.language.distribute.create_window](create_window.md)传入，不使用Window.__init__直接创建。

## 方法说明

| 方法 | 说明 |
|---|---|
| [rank](#rank) | 返回当前rank在通信组内的编号。 |
| [nrank](#nrank) | 返回通信组的rank总数。 |
| [data_ptr](#data_ptr) | 返回设备侧Window描述符的地址。 |
| [is_shared](#is_shared) | 判断窗口是否为共享窗口（每个AIV拥有独立Jetty）。 |
| [close](#close) | 释放本rank的窗口资源；建议所有rank在使用结束后共同调用。 |

### rank

返回当前rank在窗口对应通信组内的编号，取值范围为[0, nrank)。

```python
rank = win.rank()
```

### nrank

返回窗口对应通信组的rank总数。

```python
nrank = win.nrank()
```

### data_ptr

返回设备侧pypto::dist::Window描述符的地址。该地址由框架在启动Kernel时传给pypto_pro.language.distribute.Window参数，用户通常不需要直接使用。

```python
desc_addr = win.data_ptr()
```

### is_shared

判断窗口是否为共享窗口。共享窗口的所有对端共享一个本地UB_CTP端点，每个AIV拥有独立的Jetty；非共享窗口中每个对端对应一个Jetty，由多个AIV共享。

设备侧[Channel.is_shared](Channel-is_shared.md)返回同样的信息。

```python
shared = win.is_shared()
```

### close

释放本rank的窗口资源，包括设备内存、通道和屏障资源，并释放[pypto_pro.language.distribute.create_window](create_window.md)为业务Tensor保留的引用。该接口不等待其他rank，只做本地清理，尽力完成且不抛出异常。调用后不能再使用该窗口。

```python
win.close()
```

## 返回值说明

方法返回值见[方法说明](#方法说明)。

## 约束说明

- Window不能直接实例化，必须通过[pypto_pro.language.distribute.create_window](create_window.md)创建。
- create_window返回的窗口已经完成建链，无需也不能再调用连接接口。
- Window.close只释放本rank的本地资源；建议所有rank在使用结束后共同调用，避免其他rank继续访问已释放的窗口。
- 窗口的rank总数必须满足2 ≤ nrank ≤ 256。
- 窗口会持有create_window中memories传入的Tensor引用，直到Window.close被调用。
- 窗口使用期间窗口句柄必须保持有效；窗口持有的Tensor引用会随Window.close释放。

## 调用示例

```python
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import pypto_pro.language.distribute as pld


def run(group: dist.ProcessGroup):
    ws = torch.zeros(1 << 20, dtype=torch.uint8, device="npu")
    win = pld.create_window(
        group,
        name="window_demo",
        memories=[(ws, pld.Protocol.UB_MEM | pld.Protocol.URMA)],
    )

    print(win.rank(), win.nrank(), win.is_shared())
    # 在此启动使用该窗口的Kernel。

    win.close()
```
