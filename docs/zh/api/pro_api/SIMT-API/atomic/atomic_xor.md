# pypto_pro.language.simt.atomic_xor

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

以原子方式对目的操作数target与源操作数value执行按位异或，并将结果写回目的操作数。

## 函数原型

```python
pypto_pro.language.simt.atomic_xor(
    target: Union[Tile, Tensor],
    offset: Sequence[int],
    value: Union[int, float],
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| target | 输入 | 目的容器，传入Tile或Tensor，通过offset指定要更新的元素。<br>- UB Tile：必须位于UB，使用ND，支持DT_INT32、DT_UINT32。<br>- GM Tensor：必须为ND，支持DT_INT32、DT_UINT32、DT_INT64、DT_UINT64。 |
| offset | 输入 | 元素坐标序列，每个轴对应一个非负整数标量，例如[0, col]。坐标数量必须与target的维数一致，静态坐标必须在对应维度范围内。 |
| value | 输入 | 源操作数，整型常量或运行时整型标量表达式，表示按位异或掩码。数据类型必须与target一致；整数字面量按target的数据类型处理。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回更新前的target值，返回值类型与target一致。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def compute_hit_parity(
    hit: pl.Tensor[[1, 256], pl.DT_UINT32],
    parity: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    if hit[0, tid] != 0:
        pl.simt.atomic_xor(parity, [0, 0], 1)


@pl.jit()
def atomic_xor_kernel(
    hit: pl.Tensor[[1, 256], pl.DT_UINT32],
    parity: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    with pl.section_vector():
        compute_hit_parity[256](hit, parity)
```
