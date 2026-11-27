# pypto_pro.language.distribute.Channel.remote_ptr

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

返回通道内已注册区域在对端的URMA地址。传入Ptr时按其在区域内的偏移解析对端的同一字节；传入Tensor时按区域基址解析。

返回的地址是Jetty WQE使用的对端虚拟地址，不能直接传给[pypto_pro.language.load](../SIMD-API/memory_data_movement/load.md)或[pypto_pro.language.store](../SIMD-API/memory_data_movement/store.md)解引用。需要在UB_MEM平面访问对端时使用[pypto_pro.language.distribute.remote_ptr](remote_ptr.md)。

## 函数原型

```python
pypto_pro.language.distribute.Channel.remote_ptr(tensor: Tensor | Ptr) -> Ptr
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| tensor | 输入 | 窗口内注册在Protocol.URMA平面的Tensor参数，或该区域内部的[pypto_pro.language.Ptr](../SIMD-API/basic_data_structures/Ptr.md)。只解析注册在Protocol.URMA平面的区域；未注册的对象或区域外的偏移会触发错误。 |

## 返回值说明

返回对端URMA地址（[pypto_pro.language.Ptr](../SIMD-API/basic_data_structures/Ptr.md)），元素类型与传入的Tensor或Ptr一致。

## 约束说明

- 返回的地址是URMA平面地址，不能直接传给[pypto_pro.language.load](../SIMD-API/memory_data_movement/load.md)或[pypto_pro.language.store](../SIMD-API/memory_data_movement/store.md)解引用。
- 返回值可以继续用[pypto_pro.language.addptr](../SIMD-API/resource_management/addptr.md)偏移，偏移按元素个数缩放；也可以传给[write](Channel-write.md)、[read](Channel-read.md)、[write_reduce](Channel-write_reduce.md)、[atomic_fetch_add](Channel-atomic_fetch_add.md)、[atomic_cas](Channel-atomic_cas.md)和[write_notify](Channel-write_notify.md)。
- 建议只解析一次并复用结果，避免在热路径中重复解析。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_remote_ptr_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                              src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()

        ch = pld.channel(win, 0x2000, peer=(me + 1) % nr)
        # 对同一个对端地址重复使用时只解析一次。
        dst = ch.remote_ptr(ws)
        ch.write(dst, src)
        ch.write(pl.addptr(dst, 64), src)
        ch.drain()
```
