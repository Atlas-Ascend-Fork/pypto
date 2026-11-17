# 标量原子操作概述

标量原子操作对Global Memory中单个元素执行原子的读-改-写，返回更新前的旧值。多个核同时更新同一地址时不会相互覆盖，适用于多核累加统计量、维护共享计数器等场景。

与SIMT核内的原子操作不同，这里的接口在SIMD kernel（`with pypto_pro.language.section_vector():`或`with pypto_pro.language.section_cube():`）中直接调用，通过`offset`线性下标指定目标元素，与[getval](../transpose_and_element_access/getval.md)、[setval](../transpose_and_element_access/setval.md)的定位方式一致。这些接口是PIPE_S标量指令，在AI Core（AIC，Cube段）和AI Vector Core（AIV，Vector段）上均可执行。

## 公共约束

- `tensor`必须是GM Tensor（kernel参数或`pypto_pro.language.tensor_view`/`pypto_pro.language.make_tensor`视图），不支持UB Tile；SIMT函数内的UB原子操作请使用`pypto_pro.language.simt.atomic_*`。
- 标量原子操作绕过DCache。与普通标量读写混用时，需调用[pypto_pro.language.system.dcci](../cache_control/dcci.md)保证一致性。
- [atomic_store](atomic_store.md)底层为posted write，不发射`dsb`排空写缓冲；本核后续访存或跨核可见性需由调用方显式插入`dsb`或同步接口建立先后关系。
- 不同核对同一Cache Line中的不同元素执行普通标量写时会相互覆盖。多核结果应通过原子操作发布，或保证同一Cache Line只由一个核写入。
- 原子操作是PIPE_S标量指令，不保证与MTE搬运或其他核的访问顺序，需要时显式插入同步。
- 接口返回值与`tensor`的元素类型一致。
- 可作为独立语句调用以丢弃旧值。

## 轮询延时：NOP

[pypto_pro.language.system.nop](../synchronization/nop.md)（SIMD kernel，AIC/AIV均可用）和[pypto_pro.language.simt.nop](../../SIMT-API/synchronization/nop.md)（SIMT函数）请求约`n`个周期的延时，`n`必须是编译期非负整数常量，且不超过`INT32_MAX`（2147483647）。

NOP主要用于控制指令执行时序、降低轮询循环中的访问压力，并有助于规避多个SIMT线程或者多个AiCore并发读写同一地址时的潜在死锁。NOP**不是**同步或内存序原语：它不提供可见性或顺序保证，也不能替代`PipeBarrier`、`set_flag`/`wait_flag`、`dsb`或`dcci`。

对GM中的标志位、原子值或队列状态做自旋等待时，可在循环体插入NOP降低重复访问频率：

```python
while pl.atomic_load(flag, 0) != expected:
    pl.system.nop(16)
```

SIMT函数内的同等写法为`pypto_pro.language.simt.nop(16)`。两种接口的`n`均表示请求的延时周期数。

| 场景 | 建议 |
| --- | --- |
| SIMT多线程读写同一地址的竞争、潜在死锁规避 | 推荐 |
| 自旋等待/轮询循环（flag、原子值、CQ等） | 推荐 |
| 性能实验中的受控延时 | 一般 |
| 有明确时序要求的固定延时 | 有条件 |
| 内存可见性/访问顺序 | 不可用，改用`dcci`/`dsb`或同步接口 |
| 流水线同步 | 不可用，改用`PipeBarrier`、`set_flag`/`wait_flag` |

固定NOP数量对短等待和长等待都可能不是最优，应按需调整`n`。
