# Kernel核函数创建

AI CPU Kernel通过`@pypto_pro.language.jit(mode="aicpu_kernel")`声明，在函数签名中定义输入、输出Tensor和运行时Scalar参数，函数体使用标量编程面描述串行逻辑。编程模型与执行特点参见[AI CPU编程范式](../programming_paradigm/AICPU/programming_paradigm.md)。

## 参数

| 参数种类 | 支持情况 |
| --- | --- |
| Tensor参数 | 支持。使用pypto_pro.language.Tensor声明，维度支持pypto_pro.language.DYNAMIC；方向标注（pypto_pro.language.Input、pypto_pro.language.Output）与AI Core Kernel一致。 |
| Scalar参数 | 支持。以数据类型标注声明（如pypto_pro.language.DT_INT32、pypto_pro.language.DT_INT64），运行时按值传入。 |
| Ptr参数 | 支持（离线交付与Tensor等价映射）。 |
| TILING、WINDOW参数 | 不支持。JIT fast-launch与离线交付均在参数校验阶段拒绝。 |
| TilingData、TilingKey、datatype特化 | 不支持。装饰器拒绝tiling_key与datatype选项。 |

Tensor的声明语法与函数体内的读写操作参见[Tensor创建和操作](tensor_creation_and_operations.md)。

## 装饰器选项

| 选项 | 状态 | 说明 |
| --- | --- | --- |
| arch、name、compile_timeout、max_threads | 接受 | arch与name走标准代码生成与符号命名；compile_timeout约束交叉编译；max_threads为无害元数据 |
| pipeline、tiling_key、datatype、sanitizer | 拒绝 | Tile/SIMT专属机制，装饰时抛出NotSupported |
| sk_cap、schedmode | 拒绝 | SuperKernel与调度模式为AI Core机制，装饰时抛出NotSupported |
| auto_mutex | 强制关闭 | 设备侧mutex插桩在控制CPU无意义 |

## 状态返回与错误上报

AI CPU Kernel遵循状态ABI：函数体中的裸`return`表示成功（KERNEL_STATUS_OK），`return 1`表示参数非法（KERNEL_STATUS_PARAM_INVALID）。非零状态由AI CPU运行时标记任务失败，Host在同步Stream时观察到错误。

```python
import pypto_pro.language as pl


@pl.jit(mode="aicpu_kernel")
def plan_kernel(
    seq_lens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT64, pl.Input],
    plan: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT32, pl.Output],
    max_blocks: pl.DT_INT32,
):
    n = pl.numel(seq_lens)
    if n <= 0 or max_blocks <= 0:
        return 1  # KERNEL_STATUS_PARAM_INVALID
    for i in pl.range(0, n):
        pl.setval(plan, i, pl.getval(seq_lens, i) * 2)
    return  # KERNEL_STATUS_OK
```

失败任务会污染发起进程的设备上下文，验证失败路径时建议在子进程中执行，参见[AI CPU Kernel任务失败：同步时报Stream错误](../../../appendix/faq/aicpu-stream-task-failed.md)。

## 启动方式

AI CPU Kernel单任务串行执行，没有启动规模配置，Stream是唯一的方括号元素：

```python
import torch
import torch_npu


seq_lens = torch.tensor([8, 16, 32], dtype=torch.int64, device="npu")
plan = torch.zeros(3, dtype=torch.int32, device="npu")

metadata_kernel(seq_lens, plan, 8)          # 当前Stream
metadata_kernel[stream](seq_lens, plan, 8)  # 显式Stream
torch.npu.synchronize()
```

`kernel[stream, block_dim]`形式会被拒绝：AI CPU下发固定为单任务，block_dim会被校验后静默丢弃，因此前端直接报错。

## 编译与交付

AI CPU Kernel复用共享的CCE代码生成产出裸函数体（无设备限定符），再由AI CPU专用glue包装为扁平入口，经bisheng交叉编译嵌入`.aicpu_binary`段。支持两种交付：

- JIT fast-launch：首次启动时编译为共享库，通过`<op>Run` shim下发，适合开发与验证，参见[JIT编译](../development/compilation_and_execution/JIT_compilation.md)；
- 离线CpuKernel交付：生成与手写AI CPU算子同构的REGISTER_CPU_KERNEL适配代码与opInfo json，合入自定义算子包，参见[离线二进制编译](../development/compilation_and_execution/offline_binary_compilation.md#aicpu-kernel离线交付)。
