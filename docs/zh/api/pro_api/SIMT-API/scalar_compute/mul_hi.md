# pypto_pro.language.simt.mul_hi

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

计算两个同类型整数完整乘积的高半部分。对于位宽为w的输入，先按输入的有符号或无符号语义得到2w位乘积，再取高w位。

该接口不等价于先执行输入位宽的乘法再右移；后者会在乘法时丢失高位。该接口可用于多精度整数计算等场景。

## 函数原型

```python
pypto_pro.language.simt.mul_hi(
    lhs: int,
    rhs: int,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 左操作数，整型常量或运行时整型标量表达式，支持DT_INT32、DT_UINT32、DT_INT64和DT_UINT64。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 右操作数，整型常量或运行时整型标量表达式，数据类型必须与lhs一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回完整乘积的高半部分，结果为Scalar，数据类型与输入一致。

| lhs数据类型 | rhs数据类型 | 完整乘积位宽 | 返回部分 | 返回数据类型 |
|---|---|---|---|---|
| DT_INT32 | DT_INT32 | 64位 | 高32位 | DT_INT32 |
| DT_UINT32 | DT_UINT32 | 64位 | 高32位 | DT_UINT32 |
| DT_INT64 | DT_INT64 | 128位 | 高64位 | DT_INT64 |
| DT_UINT64 | DT_UINT64 | 128位 | 高64位 | DT_UINT64 |

有符号输入的结果按有符号整数解释，等价于对完整有符号乘积算术右移w位。

## 调用示例

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def multiply_high(
    lhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    output: pl.Tensor[[1, THREADS], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    lhs_value = pl.simt.cast(lhs[0, tid], pl.DT_UINT32)
    rhs_value = pl.simt.cast(rhs[0, tid], pl.DT_UINT32)
    output[0, tid] = pl.simt.mul_hi(lhs_value, rhs_value)


@pl.jit()
def simt_mul_hi_kernel(
    lhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    output: pl.Tensor[[1, THREADS], pl.DT_UINT32],
):
    with pl.section_vector():
        multiply_high[THREADS](lhs, rhs, output)
```
