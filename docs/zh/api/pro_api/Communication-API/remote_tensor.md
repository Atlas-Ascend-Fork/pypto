# pypto_pro.language.distribute.remote_tensor

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

构造对端窗口中已注册Tensor的视图，等价于对已注册区域调用[pypto_pro.language.distribute.remote_ptr](remote_ptr.md)取得指针后，再调用[pypto_pro.language.make_tensor](../SIMD-API/resource_management/make_tensor.md)构造视图。

返回的视图保持源Tensor的shape，因此元素偏移与本地Tensor一致，可以直接使用[pypto_pro.language.atomic_add](../SIMD-API/atomic/atomic_add.md)或[pypto_pro.language.atomic_load](../SIMD-API/atomic/atomic_load.md)等接口访问对端的对应元素。

## 函数原型

```python
pypto_pro.language.distribute.remote_tensor(
    win: Window,
    tensor: Tensor,
    peer: int,
) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| win | 输入 | 窗口，必须是Kernel的pypto_pro.language.distribute.Window参数。 |
| tensor | 输入 | 窗口内已注册的Tensor参数。该接口只接受Tensor参数，不能是局部Tensor或中间结果；需要从裸指针构造视图时使用[pypto_pro.language.distribute.remote_ptr](remote_ptr.md)配合[pypto_pro.language.make_tensor](../SIMD-API/resource_management/make_tensor.md)。 |
| peer | 输入 | 目的rank，可以是编译期整数，也可以是Kernel内的整型标量表达式，取值范围为[0, nrank)。peer不等于当前rank时，区域必须包含Protocol.UB_MEM，仅注册Protocol.URMA的区域会触发错误。 |

## 返回值说明

返回对端窗口中同一区域的Tensor视图，元素类型和shape与源Tensor一致。

## 约束说明

- 只依赖标量操作，只能在Kernel的Vector执行域中调用，不能在Cube执行域中调用。
- 返回视图的shape与源Tensor一致；不支持在视图中修改shape或stride。
- 通过写引擎访问对端区域时使用[Channel.remote_ptr](Channel-remote_ptr.md)。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def signal_next_kernel(win: pld.Window, sig: pl.Tensor[[256], pl.DT_INT64]):
    with pl.section_vector():
        me = win.rank()
        nr = win.nrank()
        peer = (me + 1) % nr

        # 对下一个rank的信号区第me个元素执行原子加1。
        pl.atomic_add(pld.remote_tensor(win, sig, peer), me, 1)
```
