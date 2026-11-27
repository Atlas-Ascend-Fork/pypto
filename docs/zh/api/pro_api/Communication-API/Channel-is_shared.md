# pypto_pro.language.distribute.Channel.is_shared

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

判断通道是否使用共享窗口。

## 函数原型

```python
pypto_pro.language.distribute.Channel.is_shared() -> bool
```

## 参数说明

无。

## 返回值说明

返回True表示窗口由所有对端共享，每个AIV拥有独立Jetty；返回False表示per-endpoint窗口，每个对端对应一个Jetty。

## 约束说明

- 返回的信息与Host侧[pypto_pro.language.distribute.Window.is_shared](Window.md)一致。
- 共享窗口下通道使用当前AIV自己的Jetty；per-endpoint窗口下通道使用peer对应的Jetty，由多个AIV共享。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def channel_is_shared_kernel(win: pld.Window, ws: pl.Tensor[[1024], pl.DT_UINT8],
                             src: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        # 本示例按共享窗口编写：省略peer打开通道。
        ch = pld.channel(win, 0x2000)
        shared = ch.is_shared()
        if shared:
            ch.write(ch.remote_ptr(ws), src)
            ch.drain()
```
