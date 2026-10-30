# pypto_pro.language.simt.warp_shfl_down

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

在Warp内交换标量数据，使每个Lane读取编号比自己大delta的Lane所持有的value。例如，delta=1时，Lane 0读取Lane 1的值。该接口无需通过Unified Buffer（UB）中转数据，常用于Warp内归约和相邻Lane数据复用。

width将一个Warp划分为若干连续的逻辑子组，每个子组独立交换数据。设当前Lane在子组内的逻辑Lane ID为logical_lane_id：

- 当logical_lane_id + delta < width时，读取逻辑Lane logical_lane_id + delta持有的value。
- 当logical_lane_id + delta >= width时，源Lane会越过子组上边界，此时返回当前Lane自己的value。

下图展示完整的32-Lane Warp。width=16时，线程0～15和线程16～31组成两个逻辑子组；第二行的组内Lane ID在每组中分别从0编号到15。调用warp_shfl_down(value, 1, 16)后，各组内除最后一个Lane外均读取后一个Lane的值，线程15和线程31保留自己的值。

![warp_shfl_down结果示意图](../../figures/warp_shfl_down.jpg "warp_shfl_down结果示意图")

## 函数原型

```python
pypto_pro.language.simt.warp_shfl_down(
    value: Union[int, float],
    delta: int,
    width: int = 32,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 当前Lane用于交换的标量值，支持DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。每个Lane可以传入不同的值。 |
| delta | 输入 | 相对于当前逻辑Lane向Lane ID增大方向的偏移量。整型常量或DT_UINT32类型的运行时整型标量表达式，取值范围为[0, 31]。当delta确定的目标Lane位于当前逻辑子组内时，目标Lane必须处于当前活跃线程集合中，否则读取结果未定义。 |
| width | 输入 | 逻辑子组宽度。整型常量或DT_INT32类型的运行时整型标量表达式，可取1、2、4、8、16或32，默认值为32。width小于32时，一个Warp被划分为多个连续且等宽的逻辑子组。可按位置或使用width关键字传入，但不能同时使用两种方式。 |

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

目标Lane位于当前逻辑子组内时，返回目标Lane的value；目标Lane越过子组上边界时，返回当前Lane自己的value。当delta大于等于width时，所有Lane均返回自己的value。返回数据类型与value一致。

## 调用示例

以下示例把一个Warp划分为两个16-Lane逻辑子组，使除子组边界外的Lane读取后一个Lane的值。如果values[0, tid]等于tid，则两个子组的输出分别为[1, 2, ..., 15, 15]和[17, 18, ..., 31, 31]。

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_shfl_down_example(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    warp_width = pl.simt.warp_size()
    width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl_down(values[0, tid], 1, width)


@pl.jit()
def warp_shfl_down_kernel(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        warp_shfl_down_example[WARP_SIZE](values, output)
```
