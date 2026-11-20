# pypto_pro.language.simt.erfcinv

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

计算互补误差函数的反函数，数学上满足$\operatorname{erfc}(result)=value$，返回其浮点近似值。

## 函数原型

```python
pypto_pro.language.simt.erfcinv(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 互补误差函数值，Scalar类型，仅支持DT_FP32，范围为$[0,2]$。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回互补误差函数的反函数值，数据类型为DT_FP32。特殊值如下：

| value取值 | 返回值 |
|---|---|
| 0 | +Inf |
| 2 | -Inf |
| 1 | 0 |
| 超出$[0,2]$的有限数 | NaN |
| ±Inf | NaN |
| NaN | NaN |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def erfcinv_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.erfcinv(value[0, tid])


@pl.jit()
def erfcinv_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        erfcinv_example[64](value, output)
```
