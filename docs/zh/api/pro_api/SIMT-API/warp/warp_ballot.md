# pypto_pro.language.simt.warp_ballot

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

判断当前Warp内每个活跃Lane的输入是否为真，并将判断结果汇总为一个32 bit无符号整数。

返回值中的每个bit与一个Lane一一对应：

- 第i位对应Lane i。
- Lane i处于活跃状态且predicate为真时，第i位为1。
- Lane i的predicate为假或Lane i处于非活跃状态时，第i位为0。
- 每个参与投票的Lane获得相同的32 bit位图。

warp_ballot与warp_all、warp_any的区别在于：warp_all和warp_any只返回一个整体判断结果，warp_ballot会保留每个Lane的判断结果及其位置。

## 函数原型

```python
pypto_pro.language.simt.warp_ballot(
    predicate: int,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| predicate | 输入 | 当前Lane参与投票的条件值，布尔或整型常量，或相应类型的运行时标量表达式，支持DT_BOOL和DT_INT32。值为0表示假，非0表示真。不同Lane可以传入不同的条件值。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 仅汇总执行到当前指令的活跃Lane；发生分支发散时，不同分支的参与集合不同。

## 返回值说明

返回DT_UINT32类型的条件位图。位i为1当且仅当Lane i处于活跃状态且predicate为真，非活跃Lane的对应位为0。所有参与投票的Lane返回相同结果。

## 调用示例

以下示例进行两次投票。按bit 31到bit 0排列，第一次由所有偶数Lane投赞成票，返回01010101 01010101 01010101 01010101；第二次由前16个Lane投赞成票，返回00000000 00000000 11111111 11111111。每次投票的结果都会写入所有Lane对应的输出位置。

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_ballot_example(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    output[0, tid] = pl.simt.warp_ballot((lane % 2) == 0)
    output[1, tid] = pl.simt.warp_ballot(lane < pl.simt.warp_size() // 2)


@pl.jit()
def warp_ballot_kernel(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    with pl.section_vector():
        warp_ballot_example[WARP_SIZE](output)
```
