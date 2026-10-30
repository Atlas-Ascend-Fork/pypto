# pypto_pro.language.simt.abs

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

计算源操作数的绝对值，计算公式如下：

$$result = |value|$$

## 函数原型

```python
pypto_pro.language.simt.abs(
    value: Union[int, float],
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，整型或浮点型常量，或相应类型的运行时标量表达式，支持DT_FP16、DT_BF16、DT_FP32、DT_INT32和DT_INT64。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回源操作数的绝对值，数据类型与输入一致。

对于DT_FP16和DT_BF16输入，仅清除符号位，其余位保持不变：

| value取值 | 返回值 |
|---|---|
| ±0 | +0 |
| ±Inf | +Inf |
| NaN | 清除符号位后的NaN，保留载荷及其余位 |

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def abs_fp32(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.abs(source[0, tid])


@pl.jit()
def simt_abs_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    with pl.section_vector():
        abs_fp32[256](source, output)
```
