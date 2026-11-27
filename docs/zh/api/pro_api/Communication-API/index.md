# 通信类API

通信类API基于HCCL通信域和对称窗口（Window），提供设备侧多卡通信接口。通信编程模式与完整示例参见[通信算子开发](../../../guide/programming_guide/pro/communication_development/index.md)。

```{toctree}
:maxdepth: 1
:titlesonly:

basic_data_structures/index
create_window
dump_topology
Window
barrier
channel
Channel-remote_ptr
Channel-write
Channel-read
Channel-write_reduce
Channel-atomic_fetch_add
Channel-atomic_cas
Channel-write_notify
Channel-submit
Channel-drain
Channel-lock
Channel-unlock
Channel-is_shared
buffer_channel
BufferChannel-write
BufferChannel-write_notify
BufferChannel-submit
BufferChannel-drain
BufferChannel-read
BufferChannel-write_reduce
BufferChannel-atomic_fetch_add
BufferChannel-atomic_cas
remote_ptr
remote_tensor
```
