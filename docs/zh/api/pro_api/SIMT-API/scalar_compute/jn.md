# pypto_pro.language.simt.jn

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

计算非负整数阶第一类贝塞尔函数。

$$result = J_{order}(value)$$

## 函数原型

```python
pypto_pro.language.simt.jn(
    order: Scalar,
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| order | 输入 | 阶数，Scalar类型，仅支持DT_INT32。有效阶数为非负整数。Tensor或Tile元素需通过下标访问后传入。 |
| value | 输入 | 源操作数，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。
- order为负数时返回NaN，不进行阶数数值的编译期校验。

## 返回值说明

返回指定阶数的第一类贝塞尔函数值，数据类型为DT_FP32。order为0或1时分别与[pypto_pro.language.simt.j0](j0.md)或[pypto_pro.language.simt.j1](j1.md)一致。

| 输入情况 | 返回值 |
|---|---|
| order < 0，或value为NaN | NaN |
| order = 1，value为±0或±Inf | 与value同符号的零 |
| order ≥ 2，value为±0或±Inf | +0 |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def jn_example(
    order: pl.Tensor[[1, 64], pl.DT_INT32],
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.jn(order[0, tid], value[0, tid])


@pl.jit()
def jn_kernel(
    order: pl.Tensor[[1, 64], pl.DT_INT32],
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        jn_example[64](order, value, output)
```
