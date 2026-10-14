# pypto_pro.language.simt.warp_all

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

判断当前Warp内所有活跃Lane的输入是否均为真，可以理解为对所有活跃Lane的predicate执行一次逻辑与操作。

每个活跃Lane先独立计算自己的predicate，然后共同参与投票：

- 所有活跃Lane的predicate均为真时，投票结果为1。
- 只要有一个活跃Lane的predicate为假，投票结果就为0。
- 每个参与投票的Lane获得相同的投票结果，非活跃Lane不参与判断。

## 函数原型

```python
pypto_pro.language.simt.warp_all(
    predicate: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| predicate | 输入 | 当前Lane参与投票的条件值，Scalar类型，支持DT_BOOL和DT_INT32。值为0表示假，非0表示真。不同Lane可以传入不同的条件值。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 仅汇总执行到当前指令的活跃Lane；发生分支发散时，不同分支的参与集合不同。

## 返回值说明

返回DT_INT32类型的投票结果。如果所有活跃Lane的predicate均为真，则返回1；否则返回0。所有参与投票的Lane返回相同结果。

## 调用示例

以下示例进行两次投票。第一次投票中所有Lane的条件均为真，output第0行全部为1；第二次投票中Lane 31的条件为假，output第1行全部为0。

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_all_example(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_all(lane < warp_size)
    output[1, tid] = pl.simt.warp_all(lane < warp_size - 1)


@pl.jit()
def warp_all_kernel(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        warp_all_example[WARP_SIZE](output)
```
