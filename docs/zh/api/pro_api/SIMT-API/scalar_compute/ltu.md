# pypto_pro.language.simt.ltu

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

执行“小于或无序”比较。当lhs或rhs任一为NaN，或关系lhs < rhs成立时返回真；否则返回假。

## 函数原型

```python
pypto_pro.language.simt.ltu(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，Scalar类型，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，Scalar类型，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 两个输入必须同为DT_FP16或同为DT_BF16。

## 返回值说明

返回比较结果，数据类型为DT_BOOL。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def ltu_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_BOOL],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.ltu(lhs[0, tid], rhs[0, tid])


@pl.jit()
def ltu_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_BOOL],
):
    with pl.section_vector():
        ltu_example[64](lhs, rhs, output)
```
