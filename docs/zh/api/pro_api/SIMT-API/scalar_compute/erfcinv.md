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

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 互补误差函数值，Scalar类型，仅支持DT_FP32，实数定义域为$[0,2]$。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 定义域是数学语义说明，不进行输入数值范围的编译期校验。输入超出$[0,2]$（包括正负无穷）或为NaN时返回NaN。

## 返回值说明

返回互补误差函数的反函数值，数据类型为DT_FP32。输入为0和2时分别返回正无穷和负无穷，输入为1时返回零。

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
