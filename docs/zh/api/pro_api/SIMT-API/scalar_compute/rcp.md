# pypto_pro.language.simt.rcp

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

计算value的倒数，计算公式如下。

$$result = \frac{1}{value}$$

## 函数原型

```python
pypto_pro.language.simt.rcp(
    value: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，支持DT_FP16和DT_BF16。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT函数中调用。

## 返回值说明

返回value的倒数，数据类型与输入一致。特殊值如下：

| value取值 | 返回值 |
|---|---|
| +0 | +Inf |
| -0 | -Inf |
| +Inf | +0 |
| -Inf | -0 |
| NaN | NaN |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def rcp_example(
    value: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.rcp(value[0, tid])


@pl.jit()
def rcp_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    with pl.section_vector():
        rcp_example[64](value, output)
```
