# pypto_pro.language.simt.nop

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

SIMT延时指令，在每个SIMT线程中插入约`n`个周期的空操作（NOP）。

本接口接受周期预算`n`，实际延时按目标实现的NOP粒度向上取整，可能略大于`n`。

NOP用于控制执行时序、降低自旋等待的访问压力，并有助于规避多个SIMT线程或SIMT线程与其他核并发读写同一地址时的潜在死锁。NOP不是同步或内存序原语：它不提供内存可见性或访问顺序保证，也不能替代[syncthreads](syncthreads.md)或[threadfence](threadfence.md)。

## 函数原型

```python
pypto_pro.language.simt.nop(
    n: int,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| n | 输入 | 延时周期数。编译期非负整数常量，取值范围0 ≤ n ≤ INT32_MAX（2147483647）。 |

## 约束说明

- `n`必须是编译期常量（整数字面量或可常量折叠的表达式），不支持运行时标量；为负数或超过INT32_MAX时在解析阶段报错。
- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用；SIMD kernel中的等价接口为[pypto_pro.language.system.nop](../../SIMD-API/synchronization/nop.md)。
- 必须作为独立语句调用，不支持使用其结果参与表达式。
- 固定延时对短等待与长等待都可能不是最优，应按需调整`n`。
- NOP只延时，不提供同步或内存序保证。

## 返回值说明

无。

## 调用示例

轮询GM中的标志位时，在循环体插入NOP降低重复访问频率：

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=32)
def wait_for_flag(
    flag: pl.Tensor[[1, 1], pl.DT_INT32],
    expected: pl.DT_INT32,
):
    # atomic_add(..., 0) 读取旧值且不修改标志位。
    while pl.simt.atomic_add(flag, [0, 0], 0) != expected:
        pl.simt.nop(16)


@pl.jit
def simt_nop_polling_kernel(
    flag: pl.Tensor[[1, 1], pl.DT_INT32],
    expected: pl.DT_INT32,
):
    with pl.section_vector():
        wait_for_flag[32](flag, expected)
```

更多用法见[标量原子操作概述](../../SIMD-API/atomic/atomic_overview.md)。
