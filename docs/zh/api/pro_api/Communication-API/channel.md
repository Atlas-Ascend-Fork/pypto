# pypto_pro.language.distribute.channel

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

在HCCL通信域的窗口上打开一个写引擎通道。通道绑定一个UB暂存区，用于向对端提交异步写、读、归约和原子操作。通道在打开时确定目的端：指定peer时使用该对端的Jetty（per-endpoint形式）；省略peer时使用当前AIV自己的Jetty（共享窗口形式）。通道打开后，write和write_notify等方法不再接收peer参数。

Channel的方法由解析器拦截并转换为设备侧操作，在Kernel外调用会报错。通道必须作为简单赋值的右值创建，不能放在控制流内部。

## 函数原型

```python
pypto_pro.language.distribute.channel(
    win: Window,
    buffer: int,
    /,
    *,
    peer: int | None = None,
) -> Channel
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| win | 输入 | 窗口，必须是Kernel的pypto_pro.language.distribute.Window参数。窗口必须至少注册了一个Protocol.URMA区域。 |
| buffer | 输入 | UB暂存区地址，编译期整数。必须按64字节对齐，至少为pypto_pro.language.distribute.CHANNEL_UB_BYTES（256）字节，且不能与Tile或其他通道的暂存区重叠。前128字节用于存放对端和Jetty头部信息。 |
| peer | 输入 | 目的rank，取值范围为[0, nrank)，可以是编译期整数，也可以是Kernel内的整型标量表达式。指定时使用per-endpoint形式；省略时使用共享窗口形式。peer不能等于当前rank（仅per-endpoint窗口），否则会解析到错误的Jetty。 |

## 返回值说明

返回pypto_pro.language.distribute.Channel解析期句柄，用于在Kernel的Vector执行域内调用写引擎方法。句柄字段如下：

| 字段 | 说明 |
|---|---|
| window | 通道所属的窗口。 |
| ub_addr | 通道UB暂存区地址。 |
| token | 解析器生成的通道标识。 |

## 约束说明

- 只依赖标量操作，只能在Kernel的Vector执行域中创建和使用，不能作为普通Python对象构造或保存到主机侧。
- 必须是简单赋值的右值，不能放在控制流内部。通道对象存放在外层作用域：

  ```python
  ch = pypto_pro.language.distribute.channel(win, 0x2000, peer=peer)  # 正确
  ```

- 窗口至少要有一个Protocol.URMA区域；没有URMA区域时打开通道会触发错误。
- 共享窗口（win.is_shared()为True）必须省略peer，传入的peer会在运行时触发错误；per-endpoint窗口必须指定peer，省略时同样会在运行时触发错误。
- per-endpoint窗口下peer不能等于win.rank()，否则会解析到错误的Jetty，行为未定义。
- buffer必须是编译期常量，且不能与Tile或其他通道的暂存区重叠；框架会按`buffer + pypto_pro.language.distribute.CHANNEL_UB_BYTES`预留UB空间。
- 所有timeout、submit和option参数必须是编译期常量。
- 所有远端地址必须由[Channel.remote_ptr](Channel-remote_ptr.md)解析得到，并保持在同一URMA区域内；不要把[pypto_pro.language.distribute.remote_ptr](remote_ptr.md)返回的UB_MEM指针传给写引擎方法。
- 非共享窗口中，多个AIV共享同一Jetty时要使用[Channel.lock](Channel-lock.md)和[Channel.unlock](Channel-unlock.md)保护完整的提交流水。
- 通道打开后，write、read、write_reduce、write_notify和原子操作不再接收peer参数；需要切换目的端时应重新打开通道。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_write_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                         src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        # 打开到下一个rank的通道，UB暂存区为[0x2000, 0x2100)。
        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)

        # 解析源区域在对端的URMA地址，提交64字节异步写并等待完成。
        dst = ch.remote_ptr(ws)
        ch.write(dst, src)
        ch.drain()
```
