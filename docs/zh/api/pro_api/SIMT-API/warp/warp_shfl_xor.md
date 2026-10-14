# pypto_pro.language.simt.warp_shfl_xor

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

在Warp内交换标量数据。每个Lane将自己的逻辑Lane ID与lane_mask按位异或，并读取异或结果所对应Lane持有的value。该接口无需通过Unified Buffer（UB）中转数据，常用于相邻Lane成对交换和蝶形通信。

width将一个Warp划分为若干连续的逻辑子组，每个子组独立交换数据。设当前Lane在子组内的逻辑Lane ID为logical_lane_id，则源Lane的逻辑Lane ID为：

源Lane逻辑Lane ID = logical_lane_id ^ lane_mask

其中，^表示按位异或。例如，lane_mask=1时，最低位翻转，Lane 0与Lane 1、Lane 2与Lane 3依次成对交换数据；lane_mask=2时，Lane 0与Lane 2、Lane 1与Lane 3依次交换数据。

下图展示完整的32-Lane Warp。width=16时，线程0～15和线程16～31组成两个逻辑子组；第二行的组内Lane ID在每组中分别从0编号到15。调用warp_shfl_xor(value, 1, 16)后，每组内相邻两个Lane成对交换数据。

![warp_shfl_xor结果示意图](../../figures/warp_shfl_xor.jpg "warp_shfl_xor结果示意图")

## 函数原型

```python
pypto_pro.language.simt.warp_shfl_xor(
    value: Union[int, float],
    lane_mask: int,
    width: int = 32,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 当前Lane用于交换的标量值，支持DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。每个Lane可以传入不同的值。 |
| lane_mask | 输入 | 与当前逻辑Lane ID执行按位异或的掩码。整型常量或DT_INT32类型的运行时整型标量表达式，取值范围为[0, 31]，并且必须小于width。异或后选中的目标Lane必须处于当前活跃线程集合中，否则读取结果未定义。 |
| width | 输入 | 逻辑子组宽度。整型常量或DT_INT32类型的运行时整型标量表达式，可取1、2、4、8、16或32，默认值为32。width小于32时，一个Warp被划分为多个连续且等宽的逻辑子组。可按位置或使用width关键字传入，但不能同时使用两种方式。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回当前逻辑Lane ID与lane_mask按位异或后所选Lane的value，返回数据类型与value一致。

## 调用示例

以下示例把一个Warp划分为两个16-Lane逻辑子组，并使用lane_mask=1交换每对相邻Lane的数据。如果values[0, tid]等于tid，则输出为[1, 0, 3, 2, ..., 15, 14, 17, 16, ..., 31, 30]。

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_shfl_xor_example(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    warp_width = pl.simt.warp_size()
    width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl_xor(values[0, tid], 1, width)


@pl.jit()
def warp_shfl_xor_kernel(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        warp_shfl_xor_example[WARP_SIZE](values, output)
```
