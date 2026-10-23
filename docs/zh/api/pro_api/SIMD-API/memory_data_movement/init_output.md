# pypto_pro.language.init_output

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

将GM（Global Memory）Tensor的指定区域初始化为标量值。框架内部先在UB上通过`TEXPANDS`将标量填充到临时缓冲区（V流水），再通过`TSTORE`将数据从UB搬写到GM（MTE3流水），并自动插入V→MTE3同步事件。该接口主要用于预处理场景，即在Tile缓冲区分配之前对workspace等GM内存进行批量初始化或清零，无需用户手动创建和管理Tile。

与[pypto_pro.language.expands](../tile_computation/math_functions/expands.md)的区别：expands填充的是片上Tile（UB/L1），需要用户先创建Tile并管理UB地址；init_output直接初始化GM Tensor，由框架自动分配临时UB缓冲区，适合大段workspace的批量初始化。

## 函数原型

```python
pypto_pro.language.init_output(
    tensor: Tensor,
    *,
    offset: int = 0,
    size: int,
    value: Union[int, float] = 0,
) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| tensor | 输出 | 目的操作数，Tensor类型，存储空间为GM。支持的数据类型为DT_UINT8、DT_INT8、DT_UINT16、DT_INT16、DT_BF16、DT_FP16、DT_UINT32、DT_INT32、DT_FP32、DT_UINT64、DT_INT64；layout支持ND、NZ，offset和size按Tensor的物理元素计数，layout不影响填充语义。 |
| offset | 输入 | 可选，元素级偏移，整型常量或运行时整型标量表达式，表示从Tensor起始地址开始的偏移量，单位为元素个数，默认`0`。<br>- 非负整数，不支持浮点。<br>- 与size之和不得超过Tensor总元素数。<br>- 支持传入运行时标量（如循环变量、shape表达式）。 |
| size | 输入 | 待初始化的元素个数，整型常量或运行时整型标量表达式，单位为元素个数。<br>- 正整数，不支持浮点。<br>- 支持传入运行时标量。 |
| value | 输入 | 可选，填充值，整型或浮点型常量，或相应类型的运行时标量表达式，默认`0`。<br>- 支持整型或浮点型常量（如`0`、`0.0`）、[pypto_pro.language.const](../../Utils-API/python_syntax_sugar/const.md)构造的标量，以及运行时标量表达式。<br>- 类型须与tensor元素类型兼容；整型字面量须能被tensor的数据类型表示。<br>- 当tensor为整型而value为浮点常量时，按向零截断转换为整型（小数部分被丢弃），与框架内其他整型目的操作数的标量转换保持一致。 |

## 约束说明

- 该接口属于Vector流水（V/MTE3）操作，须在[pypto_pro.language.section_vector](../controlflow/section_vector.md)标识的Vector执行域内调用。若在无执行域标识的Kernel中裸调用，框架会将该Kernel视为Cube与Vector共享，把同一段向量体一并编译进Cube核（AIC）目标，从而触发编译失败。
- 该接口主要用于预处理场景，须在Tile缓冲区分配（[pypto_pro.language.make_tile_group](../resource_management/make_tile_group.md)等）之前使用。
- 单核调用此接口时，接口内部不会插入UB复用相关的流水同步，且开启auto_mutex时框架也不会感知该接口内部的UB使用。如果后续操作涉及Unified Buffer（UB）的使用（如[pypto_pro.language.load](load.md)），需要用户自行设置MTE2流水等待MTE3流水（`MTE3_MTE2`）的同步：调用[pypto_pro.language.system.sync_src](../synchronization/sync_src.md)和[pypto_pro.language.system.sync_dst](../synchronization/sync_dst.md)（set_pipe为MTE3、wait_pipe为MTE2），保证接口内部对临时UB缓冲区的读取完成后，再复用该UB区域。
- 单核上连续多次调用此接口且value不一致时，需要在接口之间设置V流水等待MTE3流水（`MTE3_V`）的同步。由于该接口内部使用同一块UB作为中转空间进行值初始化，连续调用接口时若不进行流水同步，后一次写操作可能覆盖前一次未完成写入的数据，导致前一次初始化Global Memory的结果非预期。
- 当多个核调用此接口对Global Memory进行初始化时，所有核对Global Memory的初始化未必会同时结束，也可能存在核之间读后写、写后读以及写后写等数据依赖问题。**本接口内部不会插入全核同步屏障（`pipe_barrier`仅作用于调用该接口的当前核），必须在调用后显式使用[pypto_pro.language.system.sync_all](../synchronization/sync_all.md)（或等价的跨核同步原语）保证初始化结果对其他核（含cube子核）可见**，再执行后续操作。例如：init_output后跟matmul等cube运算时，需要在AIV侧与AIC侧分别调用`system.sync_all(core_type=SyncCoreType.MIX)`。

## 返回值说明

无。

## 调用示例

### 基本用法

下面是一个完整Kernel：用pypto_pro.language.init_output将FP32 GM Tensor的全部元素初始化为`0.0`。Vector Kernel开启auto_mutex，同步由[pypto_pro.language.make_tile_group](../resource_management/make_tile_group.md)自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def init_output_kernel(
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    with pl.section_vector():
        pl.init_output(out, offset=0, size=64 * 64, value=0.0)
```

### 分块初始化workspace

Attention等场景中常需分块初始化workspace区域。以下示例在循环中逐块将workspace的指定偏移区域清零，每块`128 * 128`个元素：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def init_dq_workspace_kernel(
    workspace: pl.Tensor[[4096 * 4096], pl.DT_FP32],
    dq_offset: pl.DT_INT32,
    init_size: pl.DT_INT32,
):
    with pl.section_vector():
        for i in pl.range(0, init_size, 128 * 128):
            pl.init_output(workspace, offset=dq_offset + i, size=128 * 128, value=0.0)
```

其他典型用法（节选）：

```python
with pl.section_vector():
    # 使用 pl.const 构造标量值
    pl.init_output(workspace, offset=dq_offset, size=128 * 128,
                   value=pl.const(0, pl.DT_FP32))

    # 初始化为非零初值（如负无穷，用于掩码workspace）
    pl.init_output(mask_workspace, offset=0, size=1024,
                   value=pl.const(float('-inf'), pl.DT_FP32))
```
