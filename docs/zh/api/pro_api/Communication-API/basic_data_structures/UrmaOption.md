# pypto_pro.language.distribute.UrmaOption

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

URMA写引擎提交工作队列元素（WQE）时的控制标志，是enum.IntFlag，多个标志可以通过按位或组合使用。

这些标志只在UB_CTP/URMA平面上有意义，在RoCE平面上只有FENCE和CQE生效。

## 原型定义

```python
class UrmaOption(enum.IntFlag):
    ODR_NONE = ...  # WQE之间没有执行顺序要求
    ODR_RELAX = ...  # 宽松执行顺序，后提交的WQE可以在前一个完成前开始；与ODR_STRONG互斥
    ODR_STRONG = ...  # 强执行顺序，WQE按提交顺序执行；与ODR_RELAX互斥
    ODR_IN_ORDER_COMPLETION = ...  # 完成记录（CQE）按提交顺序上报
    FENCE = ...  # WQE等待此前提交的所有WQE完成后再执行
    CQE = ...  # 请求一个完成记录；单独提交的Channel操作必须包含该标志
    DEFAULT = ...  # 宽松执行顺序、按序完成、带fence、每个WQE一个完成记录
```

## 参数说明

无。

## 约束说明

- option必须是编译期常量，不能是运行时表达式。
- 只能使用本枚举中列出的标志位，设置其他标志位会报错。
- ODR_RELAX和ODR_STRONG不能同时设置。
- [Channel](../channel.md)的发起类方法要求option包含CQE，默认值UrmaOption.DEFAULT已包含。
- [BufferChannel](../buffer_channel.md)的追加类方法中，调用方设置的CQE位会被忽略，由BufferChannel.submit在最后一个WQEBB上统一设置FENCE和CQE。
- 标志只在UB_CTP/URMA平面上生效；RoCE平面只识别FENCE和CQE。

## 返回值说明

无。

## 调用示例

```python
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld


@pl.jit(auto_mutex=True)
def option_example(win: pld.Window, ws: pl.Tensor[[64], pl.DT_UINT8],
                   src: pl.Tensor[[64], pl.DT_UINT8], out: pl.Tensor[[64], pl.DT_UINT8]):
    with pl.section_vector():
        ch = pld.channel(win, 0x2000, peer=(win.rank() + 1) % win.nrank())
        dst = ch.remote_ptr(ws)

        # 使用默认配置。
        ch.write(dst, src)

        # 强顺序执行，等待前序写完成，并要求完成记录。
        ch.write(dst, src, option=pld.UrmaOption.ODR_STRONG | pld.UrmaOption.FENCE | pld.UrmaOption.CQE)

```
