# pypto_pro.language.simt.isfinite

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

判断源操作数是否为有限浮点数，即既不是正负无穷，也不是NaN。

## 函数原型

```python
pypto_pro.language.simt.isfinite(
    value: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，支持DT_FP16和DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

输入为有限数时返回True，包括正负零、正规数和次正规数；输入为NaN、正无穷或负无穷时返回False。返回值为DT_BOOL类型的Scalar。

## 调用示例

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def classify(
    source: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_BOOL],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.isfinite(source[0, tid])


@pl.jit()
def simt_isfinite_kernel(
    source: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_BOOL],
):
    with pl.section_vector():
        classify[THREADS](source, output)
```
