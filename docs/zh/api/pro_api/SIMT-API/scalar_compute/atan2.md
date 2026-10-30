# pypto_pro.language.simt.atan2

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

根据y和x的符号确定象限，计算坐标$(x,y)$相对$x$轴正方向的极角。结果以弧度表示。

$$result = \operatorname{atan2}(y, x)$$

## 函数原型

```python
pypto_pro.language.simt.atan2(
    y: float,
    x: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| y | 输入 | 纵坐标，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| x | 输入 | 横坐标，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT函数中调用。

## 返回值说明

返回坐标$(x,y)$的极角，数据类型为DT_FP32，值域为$[-\pi,\pi]$。特殊值如下：

| y、x取值 | 返回值 |
|---|---|
| 任一输入为NaN | NaN |
| y为±0，x的符号位为0（包括+0） | 返回y |
| y为±0，x的符号位为1（包括-0） | 绝对值为$\pi$，符号与y一致 |
| y非零，x为±0 | 绝对值为$\frac{\pi}{2}$，符号与y一致 |
| y为±Inf，x为+Inf | 绝对值为$\frac{\pi}{4}$，符号与y一致 |
| y为±Inf，x为-Inf | 绝对值为$\frac{3\pi}{4}$，符号与y一致 |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def atan2_example(
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.atan2(y[0, tid], x[0, tid])


@pl.jit()
def atan2_kernel(
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        atan2_example[64](y, x, output)
```
