# pypto_pro.language.distribute.remote_ptr

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

返回指向对端窗口中已注册区域某一字节的UB_MEM指针。区域和偏移在运行时根据入参解析，因此指针可以指向区域内的任意位置，不要求是区域起始地址。

返回的指针具有元素类型，可直接传给[pypto_pro.language.make_tensor](../SIMD-API/resource_management/make_tensor.md)构造Tensor视图，再通过[pypto_pro.language.load](../SIMD-API/memory_data_movement/load.md)、[pypto_pro.language.store](../SIMD-API/memory_data_movement/store.md)或原子接口访问对端内存。

## 函数原型

```python
pypto_pro.language.distribute.remote_ptr(
    win: Window,
    tensor: Tensor | Ptr,
    peer: int,
) -> Ptr
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| win | 输入 | 窗口，必须是Kernel的pypto_pro.language.distribute.Window参数。 |
| tensor | 输入 | 窗口内已注册的Tensor参数，或指向已注册区域内部的[pypto_pro.language.Ptr](../SIMD-API/basic_data_structures/Ptr.md)。传入Tensor时按区域基址解析；传入Ptr时可以携带指针偏移。必须是窗口内注册的区域，未注册的Tensor或区域外的Ptr会触发错误。 |
| peer | 输入 | 目的rank，可以是编译期整数，也可以是Kernel内的整型标量表达式，取值范围为[0, nrank)。peer等于当前rank时返回本rank的窗口地址，可用于本地回环访问；peer不等于当前rank时，区域必须包含Protocol.UB_MEM，仅注册Protocol.URMA的区域会触发错误。 |

## 返回值说明

返回与源对象元素类型相同的对端指针（Ptr）。该指针可以直接构造Tensor视图，也可以继续用[pypto_pro.language.addptr](../SIMD-API/resource_management/addptr.md)偏移。

## 约束说明

- 只依赖标量操作，只能在Kernel的Vector执行域中调用，不能在Cube执行域中调用。
- 指向对端UB_MEM区域的指针只适用于[pypto_pro.language.load](../SIMD-API/memory_data_movement/load.md)、[pypto_pro.language.store](../SIMD-API/memory_data_movement/store.md)和原子接口，不能用于[pypto_pro.language.printf](../Utils-API/debugging/printf.md)等主机侧接口。
- 指针算术按元素个数缩放，不是按字节缩放。需要按字节偏移时，先使用[pypto_pro.language.make_ptr](../SIMD-API/resource_management/make_ptr.md)转换为字节指针。
- 通过写引擎访问对端区域时使用[Channel.remote_ptr](Channel-remote_ptr.md)，不能使用本接口。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld

_WS_BYTES = 1 << 20
_SHIFT_N = 1024
_DTYPE = pl.DT_BF16


@pl.jit(auto_mutex=True)
def remote_ptr_store_kernel(win: pld.Window, ws: pl.Tensor[[_WS_BYTES], pl.DT_UINT8],
                            src: pl.Tensor[[1, _SHIFT_N], _DTYPE]):
    with pl.section_vector():
        tile_type = pl.TileType(shape=[1, _SHIFT_N], dtype=_DTYPE, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tile_type, addr=0)
        pl.load(t, src, [0, 0])

        me = win.rank()
        nr = win.nrank()
        # 解析下一个rank窗口中对区域基址的指针，并写入本rank数据。
        put_view = pl.make_tensor(
            pld.remote_ptr(win, ws, (me + 1) % nr), [1, _SHIFT_N], dtype=_DTYPE)
        pl.store(put_view, t, [0, 0])
        # 对端可见性由后续信号或同步协议保证。
```
