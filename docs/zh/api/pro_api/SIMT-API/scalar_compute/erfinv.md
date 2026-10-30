# pypto_pro.language.simt.erfinv

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

计算误差函数的反函数，数学上满足$\operatorname{erf}(result)=value$，返回其浮点近似值。

## 函数原型

```python
pypto_pro.language.simt.erfinv(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 误差函数值，Scalar类型，仅支持DT_FP32，实数定义域为$[-1,1]$。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。
- 定义域是数学语义说明，不进行输入数值范围的编译期校验。输入超出$[-1,1]$（包括正负无穷）或为NaN时返回NaN。

## 返回值说明

返回误差函数的反函数值，数据类型为DT_FP32。输入为-1和1时分别返回负无穷和正无穷；正负零保留原符号。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def erfinv_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.erfinv(value[0, tid])


@pl.jit()
def erfinv_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        erfinv_example[64](value, output)
```
