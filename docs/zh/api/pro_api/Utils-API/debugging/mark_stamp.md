# pypto_pro.language.mark_stamp

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

在指定流水上插入一个带整数ID的性能打点，供msprof采集。没有调用本接口的Kernel不会生成打点指令。实际采集过程中需要使用`msprof --instr-profiling=on`开启指令级采集。
所有执行到调用位置的Block都会打点，不隐式限制Block或子核索引。
如需仅观察部分Block，可显式使用[pypto_pro.language.get_block_idx()](../../SIMD-API/system_variables/get_block_idx.md)编写条件。

## 函数原型

```python
pypto_pro.language.mark_stamp(pipe: PipeType, id: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ---- | --------- | ---- |
| pipe | 输入 | 编译期常量，取值为[pypto_pro.language.PipeType](../../SIMD-API/basic_data_structures/PipeType.md)枚举；Cube支持S、MTE2、MTE1、M、FIX、MTE3，Vector支持S、MTE2、V、MTE3，不支持ALL。|
| id | 输入 | 打点ID，用于在msprof采集记录中区分不同打点位置；须为编译期整数，取值范围为[0, 4095]，不接受布尔值。|

## 约束说明

- 仅支持编译目标3510。支持在SIMD的Cube或Vector代码中调用，不支持在@pypto_pro.language.vector_function内调用，也不支持SIMT Kernel。
- pipe和id必须在编译期确定。不接受运行时参数或循环变量作为ID，也不接受浮点数、字符串和布尔值。
  参数类型、ID范围、流水与Kernel的匹配，以及函数作用域限制均在生成设备代码前校验。
- 同一执行单元（Cube或Vector）内每一处打点（含内联函数/流水编排段每次展开后的调用）必须使用不同ID；循环体内同一处打点在运行时重复执行不受限制。Cube与Vector可以使用相同ID。
- msprof采集记录通过整数ID区分打点。本接口不接受名称，也不生成源码映射文件。
- 打点仅标记流水上的检查点，不保证所观测操作的开始/完成时间戳；建议在既有程序同步点附近稀疏打点，不要仅为对齐打点额外插入同步。
- 打点可能引入同步副作用。CANN可在Cube侧的MTE1、MTE2、MTE3及Vector侧的MTE2、MTE3打点前插入对应流水的同步屏障，等待该流水此前的操作完成；实际同步行为以所用CANN版本为准。
  此类同步会影响被观测程序的等待时间和流水重叠情况，打点开销不局限于Kernel末尾的延时。
- 打点过密时部分打点可能被采集器丢弃或合并。
- 为排空末尾打点，包含打点的Kernel会在每个Block返回前执行循环任务（会造成Kernel执行时间变长），即使运行时条件使某些Block跳过了打点。
  Kernel总耗时包含此开销；比较实际算子性能时，应使用未打点版本。

## 返回值说明

无。

## 调用示例

以下片段放在@pypto_pro.language.jit函数中，省略了Tile和Tensor的定义：

```python
import pypto_pro.language as pl

with pl.section_vector():
    pl.mark_stamp(pl.PipeType.MTE2, 0)  # 输入搬运前
    pl.load(a, x, [0, 0])
    pl.mark_stamp(pl.PipeType.MTE2, 1)  # 输入搬运后
```

如需仅让全局AIV逻辑索引为1的核打点，可在@pypto_pro.language.jit函数中显式筛选：

```python
import pypto_pro.language as pl
with pl.section_vector():
    if pl.get_block_idx() == 1:
        pl.mark_stamp(pl.PipeType.S, 2)
```

其他Block仍正常执行条件之外的计算。
