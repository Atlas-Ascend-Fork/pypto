# pypto_pro.language.simt.atan

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

计算value的反正切值，结果以弧度表示，计算公式如下。

$$result = \arctan(value)$$

## 函数原型

```python
pypto_pro.language.simt.atan(
    value: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回value的反正切值，数据类型为DT_FP32，值域为$[-\frac{\pi}{2}, \frac{\pi}{2}]$。特殊值如下：

| value取值 | 返回值 |
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | $+\frac{\pi}{2}$ |
| -Inf | $-\frac{\pi}{2}$ |
| NaN | NaN |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def atan_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.atan(value[0, tid])


@pl.jit()
def atan_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        atan_example[64](value, output)
```
