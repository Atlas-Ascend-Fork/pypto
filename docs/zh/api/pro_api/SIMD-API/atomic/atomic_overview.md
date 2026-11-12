# 标量原子操作概述

标量原子操作对Global Memory中单个元素执行原子的读-改-写，返回更新前的旧值。多个核同时更新同一地址时不会相互覆盖，适用于多核累加统计量、维护共享计数器等场景。

与SIMT核内的原子操作不同，这里的接口在SIMD kernel（`with pypto_pro.language.section_vector():`）中直接调用，通过`offset`线性下标指定目标元素，与[getval](../transpose_and_element_access/getval.md)、[setval](../transpose_and_element_access/setval.md)的定位方式一致。

## 公共约束

- `tensor`必须是GM Tensor（kernel参数或`pypto_pro.language.tensor_view`/`pypto_pro.language.make_tensor`视图），不支持UB Tile；SIMT函数内的UB原子操作请使用`pypto_pro.language.simt.atomic_*`。
- 标量原子操作绕过DCache。与普通标量读写混用时，需调用[pypto_pro.language.system.dcci](../cache_control/dcci.md)保证一致性。
- 不同核对同一Cache Line中的不同元素执行普通标量写时会相互覆盖。多核结果应通过原子操作发布，或保证同一Cache Line只由一个核写入。
- 原子操作是PIPE_S标量指令，不保证与MTE搬运或其他核的访问顺序，需要时显式插入同步。
- 接口返回值与`tensor`的元素类型一致。
- 可作为独立语句调用以丢弃旧值。
