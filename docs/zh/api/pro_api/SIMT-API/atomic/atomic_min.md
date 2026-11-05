# pypto_pro.language.simt.atomic_min

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

以原子方式比较目的操作数target的旧值与源操作数value，并将较小值写回目的操作数。对于数据类型为DT_FP16和DT_BF16的目的操作数，只更新目的操作数，不返回旧值。

## 函数原型

```python
pypto_pro.language.simt.atomic_min(
    target: Union[Tile, Tensor],
    offset: Sequence[int],
    value: Union[int, float],
) -> Optional[Scalar]
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| target | 输入 | 目的容器，传入Tile或Tensor，通过offset指定要更新的元素。<br>- UB Tile：必须位于UB，使用ND，支持DT_INT32、DT_UINT32、DT_FP16、DT_BF16、DT_FP32。<br>- GM Tensor：必须为ND，支持DT_INT32、DT_UINT32、DT_FP16、DT_BF16、DT_FP32、DT_INT64、DT_UINT64。 |
| offset | 输入 | 元素坐标序列，每个轴对应一个非负整数标量，例如[0, col]。坐标数量必须与target的维数一致，静态坐标必须在对应维度范围内。 |
| value | 输入 | 源操作数，整型或浮点型常量，或相应类型的运行时标量表达式，表示比较值。数据类型必须与target一致；数值字面量按target的数据类型处理，整数目的操作数不接受浮点字面量。 |

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

- 是否使用返回值可能影响编译器生成的原子指令。业务场景允许时，建议不使用返回值。

- target旧值和value均为非数（NaN）时，target更新为NaN。仅一方为NaN、另一方为有限值时，接口无返回值或返回值未被实际使用，则target更新为NaN；返回值被实际使用，则target更新为该有限值。

## 返回值说明

- target的数据类型为DT_FP16或DT_BF16时无返回值，只能作为独立语句调用；
- target的数据类型为支持的其他数据类型时，返回更新前的target值，返回值类型与target一致。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def reduce_min(
    latency: pl.Tensor[[1, 256], pl.DT_UINT32],
    min_latency: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    pl.simt.atomic_min(min_latency, [0, 0], latency[0, tid])


@pl.jit()
def atomic_min_kernel(
    latency: pl.Tensor[[1, 256], pl.DT_UINT32],
    min_latency: pl.Tensor[[1, 1], pl.DT_UINT32],
):
    with pl.section_vector():
        reduce_min[256](latency, min_latency)
```
