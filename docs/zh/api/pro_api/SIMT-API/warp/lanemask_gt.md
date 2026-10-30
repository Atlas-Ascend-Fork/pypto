# pypto_pro.language.simt.lanemask_gt

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

获取当前线程的一个32位掩码。在当前线程所属的Warp中，Lane ID严格大于当前线程Lane ID的所有Lane对应位为1，其余位为0。

例如，当前Lane ID为5时，按bit 31到bit 0排列，返回掩码的32位二进制位模式为11111111 11111111 11111111 11000000。图中Lane ID从左向右递增，蓝色区域表示对应位为1，橙色边框表示当前Lane。

**图1** lanemask_gt示意图

![lanemask_gt示意图](../../figures/lanemask_gt.jpg "lanemask_gt示意图")

## 函数原型

```python
pypto_pro.language.simt.lanemask_gt() -> Scalar
```

## 参数说明

无。

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回DT_INT32类型的32 bit Lane Mask。位i为1当且仅当i大于当前Lane ID。返回值只描述Lane位置，不表示Lane是否处于活跃状态。

## 调用示例

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def lanemask_gt_example(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lanemask_gt()


@pl.jit()
def lanemask_gt_kernel(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        lanemask_gt_example[WARP_SIZE](output)
```
