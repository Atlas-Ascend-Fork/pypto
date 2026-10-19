# pypto_pro.language.simt.rint

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

将源操作数舍入到最接近的整数值；源操作数位于两个整数中间时，选择偶数。

## 函数原型

```python
pypto_pro.language.simt.rint(
    value: float,
    *,
    dtype: DType | None = None,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，支持DT_FP16、DT_BF16和DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| dtype | 输入 | 可选的返回数据类型。默认值为None。可设置为与value相同的数据类型；value为DT_FP32时，还可设置为DT_INT64。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- dtype设置为DT_INT64时，value必须为DT_FP32。

## 返回值说明

返回舍入后的整数值形式。dtype省略或设置为None时，数据类型与value一致；dtype设置为DT_INT64时，返回DT_INT64标量，超出$[-2^{63},2^{63}-1]$范围的结果钳位到对应端点。

dtype设置为DT_INT64时，特殊值的转换结果如下。

| value取值 | 返回值 |
|---|---|
| ±0 | 0 |
| +Inf或超出DT_INT64上界 | $2^{63}-1$ |
| -Inf或低于DT_INT64下界 | $-2^{63}$ |
| NaN | 0 |

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def nearest_even(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output_fp32: pl.Tensor[[1, 256], pl.DT_FP32],
    output_int64: pl.Tensor[[1, 256], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    output_fp32[0, tid] = pl.simt.rint(source[0, tid])
    output_int64[0, tid] = pl.simt.rint(source[0, tid], dtype=pl.DT_INT64)


@pl.jit()
def simt_rint_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output_fp32: pl.Tensor[[1, 256], pl.DT_FP32],
    output_int64: pl.Tensor[[1, 256], pl.DT_INT64],
):
    with pl.section_vector():
        nearest_even[256](source, output_fp32, output_int64)
```
