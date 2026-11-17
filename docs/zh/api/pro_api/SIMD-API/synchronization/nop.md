# pypto_pro.language.system.nop

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

延时指令，向当前指令流插入约`n`个周期的空操作（NOP）。

NOP用于控制指令执行时序、降低轮询循环中的访问压力，并有助于规避多个核并发读写同一地址时的潜在死锁。NOP不是同步或内存序原语：它不提供内存可见性或访问顺序保证，也不能替代[dsb](dsb.md)、bar_*或set_cross_core/wait_cross_core等同步接口。

## 函数原型

```python
pypto_pro.language.system.nop(
    n: int,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| n | 输入 | 延时周期数。编译期非负整数常量，取值范围0 ≤ n ≤ INT32_MAX（2147483647）。 |

## 约束说明

- `n`必须是编译期常量（整数字面量或可常量折叠的表达式），不支持运行时标量；为负数或超过INT32_MAX时在解析阶段报错。
- 只能在SIMD kernel（`with pl.section_vector():`或`with pl.section_cube():`）中调用，AIC与AIV均可执行；SIMT函数内请使用[pypto_pro.language.simt.nop](../../SIMT-API/synchronization/nop.md)。
- 必须作为独立语句调用，不支持使用其结果参与表达式。
- NOP只延时，不提供同步或内存序保证；等待其他核或异步pipe时仍需使用对应的同步接口。

## 返回值说明

无。

## 调用示例

轮询GM中的标志位时，在循环体插入NOP降低重复访问频率：

```python
import pypto_pro.language as pl


@pl.jit
def nop_polling_kernel(
    flag: pl.Tensor[[1], pl.DT_INT32],
    expected: pl.DT_INT32,
):
    with pl.section_vector():
        while pl.atomic_load(flag, 0) != expected:
            pl.system.nop(16)
```

更多用法见[标量原子操作概述](../atomic/atomic_overview.md)。
