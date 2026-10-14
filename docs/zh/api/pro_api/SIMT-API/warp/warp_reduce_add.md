# pypto_pro.language.simt.warp_reduce_add

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

对当前Warp内所有活跃Lane的value求和，Warp内所有活跃线程返回相同的结果。

## 函数原型

```python
pypto_pro.language.simt.warp_reduce_add(
    value: Union[int, float],
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 当前Lane参与归约的标量值，支持DT_INT32、DT_UINT32、DT_FP16和DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 仅归约执行到当前指令的活跃Lane；发生分支发散时，不同分支的参与集合不同。
- 当数据求和结果溢出时，本接口不保证计算精度。
- 本接口底层使用二分（树形）归约算法，计算顺序可能与顺序累加不同。例如，(((a + b) + c) + d)与((a + b) + (c + d))的计算顺序不同。浮点加法使用有限精度表示且每步可能发生舍入，因此不同的归约顺序可能产生不同的中间结果和最终结果。

## 返回值说明

返回所有活跃Lane的value之和，数据类型与value一致。所有活跃Lane返回相同结果。

## 调用示例

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_reduce_add_example(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.warp_reduce_add(values[0, tid])


@pl.jit()
def warp_reduce_add_kernel(
    values: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
    output: pl.Tensor[[1, WARP_SIZE], pl.DT_FP32],
):
    with pl.section_vector():
        warp_reduce_add_example[WARP_SIZE](values, output)
```
