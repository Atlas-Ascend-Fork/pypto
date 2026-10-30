# pypto_pro.language.simt.popcount

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

统计源操作数二进制表示中值为1的比特位数量。可用于位图计数，或与按位异或组合计算两个整数之间的汉明距离。

## 函数原型

```python
pypto_pro.language.simt.popcount(
    value: int,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，整型常量或运行时整型标量表达式，支持DT_UINT32和DT_UINT64。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回值为DT_INT32类型的Scalar。不同输入类型的统计范围如下：

| value数据类型 | 统计范围 | 返回值范围 |
|---|---|---|
| DT_UINT32 | 全部32个比特位 | [0, 32] |
| DT_UINT64 | 全部64个比特位，包括高32位 | [0, 64] |

## 调用示例

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def count_bits(
    source: pl.Tensor[[1, THREADS], pl.DT_UINT64],
    output: pl.Tensor[[1, THREADS], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.popcount(source[0, tid])


@pl.jit()
def simt_popcount_kernel(
    source: pl.Tensor[[1, THREADS], pl.DT_UINT64],
    output: pl.Tensor[[1, THREADS], pl.DT_INT32],
):
    with pl.section_vector():
        count_bits[THREADS](source, output)
```
