# pypto_pro.language.simt.warp_active_mask

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

获取当前指令位置上Warp的活跃Lane集合，并将其编码为一个32 bit无符号整数。该接口不判断业务条件，而是直接记录哪些Lane正在执行当前指令。

返回值中的每个bit与一个Lane一一对应：

- 第i位对应Lane i。
- Lane i正在执行当前指令时，第i位为1；否则为0。
- 执行到同一调用位置的活跃Lane获得相同的位图。

活跃Lane集合与调用位置有关。发生分支发散时，每条分支路径只包含当前正在执行该路径的Lane；各路径汇合后，重新执行同一指令的Lane会再次包含在活跃集合中。

## 函数原型

```python
pypto_pro.language.simt.warp_active_mask() -> Scalar
```

## 参数说明

无。

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回DT_UINT32类型的活跃Lane位图。位i为1表示Lane i在当前指令位置处于活跃状态。执行到同一调用位置的活跃Lane返回相同结果。

## 调用示例

以下示例让32个Lane组成的Warp的前半部分和后半部分进入不同分支。按bit 31到bit 0排列，前半部分Lane执行第一个调用位置时得到00000000 00000000 11111111 11111111，后半部分Lane执行第二个调用位置时得到11111111 11111111 00000000 00000000。

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_active_mask_example(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    if lane < pl.simt.warp_size() // 2:
        output[0, tid] = pl.simt.warp_active_mask()
        output[1, tid] = 0
    else:
        output[0, tid] = 0
        output[1, tid] = pl.simt.warp_active_mask()


@pl.jit()
def warp_active_mask_kernel(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    with pl.section_vector():
        warp_active_mask_example[WARP_SIZE](output)
```
