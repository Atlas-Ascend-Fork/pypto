# pypto_pro.language.simt.saturate

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

将FP32标量钳位到$[0,1]$，并将NaN映射为零。

## 函数原型

```python
pypto_pro.language.simt.saturate(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回DT_FP32类型的Scalar，取值范围为$[0,1]$。

| value取值 | 返回值 |
|---|---|
| value ≤ 0（包括-Inf） | 0 |
| 0 < value < 1 | value |
| value ≥ 1（包括+Inf） | 1 |
| NaN | 0 |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def saturate_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.saturate(value[0, tid])


@pl.jit()
def saturate_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        saturate_example[64](value, output)
```
