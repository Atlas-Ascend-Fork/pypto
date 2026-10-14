# pypto_pro.language.simt.warp_shfl

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

在Warp内交换标量数据，使每个Lane可以直接读取指定Lane持有的value，无需通过Unified Buffer（UB）中转数据。该接口常用于将某个Lane的数据广播给逻辑子组内的其他Lane，或让不同Lane按各自的src_lane读取数据。

width将一个Warp划分为若干连续的逻辑子组，每个子组独立交换数据。例如，width=8时，一个32-Lane的Warp被划分为4个子组：Lane 0～7、Lane 8～15、Lane 16～23和Lane 24～31。每个子组内的逻辑Lane ID都从0重新编号。

对于当前Lane，源Lane的位置为：

源Lane位置 = 子组起始Lane ID + (src_lane % width)

因此，相同的src_lane会在每个逻辑子组内选择相同的相对位置，而不是固定选择Warp中的同一个物理Lane。

下图展示完整的32-Lane Warp。width=16时，线程0～15和线程16～31组成两个逻辑子组；第二行的组内Lane ID在每组中分别从0编号到15。调用warp_shfl(value, 3, 16)后，两组分别读取线程3和线程19持有的值。

![warp_shfl结果示意图](../../figures/warp_shfl.jpg "warp_shfl结果示意图")

## 函数原型

```python
pypto_pro.language.simt.warp_shfl(
    value: Union[int, float],
    src_lane: int,
    width: int = 32,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 当前Lane用于交换的标量值，支持DT_INT32、DT_UINT32、DT_INT64、DT_UINT64、DT_FP16、DT_BF16和DT_FP32。每个Lane可以传入不同的值。 |
| src_lane | 输入 | 源Lane在逻辑子组内的编号。整型常量或DT_INT32类型的运行时整型标量表达式，取值范围为[0, 31]。当src_lane大于等于width时，使用src_lane % width得到实际的逻辑Lane ID。不同Lane可以传入不同的src_lane。src_lane指定的目标Lane必须处于当前活跃线程集合中，否则读取结果未定义。 |
| width | 输入 | 逻辑子组宽度。整型常量或DT_INT32类型的运行时整型标量表达式，可取1、2、4、8、16或32，默认值为32。width小于32时，一个Warp被划分为多个连续且等宽的逻辑子组。可按位置或使用width关键字传入，但不能同时使用两种方式。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 数据交换只发生在当前Warp和当前逻辑子组内。

## 返回值说明

返回当前逻辑子组内指定Lane的value，返回数据类型与value一致。

## 调用示例

以下示例把一个Warp划分为两个16-Lane逻辑子组，并在各子组内广播逻辑Lane 3的值。如果values[0, tid]等于tid，则Lane 0～15的输出均为3，Lane 16～31的输出均为19。

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_shfl_example(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    warp_width = pl.simt.warp_size()
    width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    output[0, tid] = pl.simt.warp_shfl(values[0, tid], 3, width)


@pl.jit()
def warp_shfl_kernel(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        warp_shfl_example[WARP_SIZE](values, output)
```
