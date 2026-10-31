# SuperKernel

## 功能说明

SuperKernel是一种算子二进制融合技术：图编译阶段把多个已编译的子核函数融合成一个超级核函数，减少任务调度与下发开销；融合后的子核共享一次下发，相邻子核之间通过核间同步事件完成依赖交接。

PyPTO Pro通过pypto_pro.language.jit的sk_cap参数声明Kernel的SK能力。设置后，JIT在Host侧Launcher中额外生成`__sk__`入口与`SK_BIND`记录，并在编译阶段追加SK相关的头文件路径和编译宏；kernel.cpp内容与普通构建一致。Kernel是否被融合由图层决定：未融合时按普通Kernel执行，早启接口不产生效果。

## 使用方法

### 声明SK能力（sk_cap）

在pypto_pro.language.jit中通过sk_cap声明能力位，取值为pypto_pro.language.SkCap的成员或其按位或组合，只接受枚举成员，不接受裸整数。各能力位的位值与含义见[JIT编译：JIT装饰器参数](../development/compilation_and_execution/JIT_compilation.md#jit装饰器参数)。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
def exp_tile_kernel(input_tensor, output_tensor):
    ...
```

### 在torch.compile中开启SuperKernel <a id="enable-superkernel"></a>

图编译侧还需开启SuperKernel优化。通过npugraph_ex后端编译时，在torch.compile的options中设置super_kernel_optimize=True：

```python
model = torch.compile(
    module,
    backend="npugraph_ex",
    options={"super_kernel_optimize": True},
)
```

如需控制融合范围，可通过torch.npu.super_kernel_scope_begin与torch.npu.super_kernel_scope_end一对接口标定参与融合的子图：两个接口成对使用、scope标识一致，二者之间标记为同一scope的算子作为一个融合范围参与SuperKernel。

```python
class Model(torch.nn.Module):
    def forward(self, x):
        torch.npu.super_kernel_scope_begin("sk_scope")
        y = exp_tile_op(x)  # 参与融合的PyPTO Pro子核
        y = y + 1.0         # 同范围内的torch算子一并参与融合
        torch.npu.super_kernel_scope_end("sk_scope")
        return y


model = torch.compile(
    Model(),
    backend="npugraph_ex",
    options={"super_kernel_optimize": True},
)
```

### 早启（Early-Start）交互 <a id="early-start"></a>

相邻子核之间存在可重叠的依赖时，通过成对的早启接口让指定位置的指令与相邻子核并行：

| 接口                                          | 作用                                                             |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| pypto_pro.language.system.wait_pre_task_end   | 调用点之前的指令可与前序子核并行，配合CAP_1_EARLY_START_WAIT使用 |
| pypto_pro.language.system.set_next_task_start | 调用点之后的指令可与后序子核并行，配合CAP_2_EARLY_START_SET使用  |

- 两个接口作用于同一个子核间边界的两侧，应在相邻子核中配合使用：前序子核调用set_next_task_start，后序子核调用wait_pre_task_end。SK图编译按相邻子核配对生成同步任务，未配对时同步任务退化为空同步，不产生重叠效果。
- 每个核、每个TilingKey分支中必须且只能调用一次，避免在运行时if/for中按不同条件调用；所有可能运行的TilingKey分支都应调用，否则同步指令数量不匹配可能卡住。
- force参数为编译期bool常量，默认True；SK子核上下文中非强制形式会被编译优化掉。

详细说明见[pypto_pro.language.system.wait_pre_task_end](../../../../api/pro_api/SIMD-API/synchronization/wait_pre_task_end.md)与[pypto_pro.language.system.set_next_task_start](../../../../api/pro_api/SIMD-API/synchronization/set_next_task_start.md)。

### 调度模式（schedmode）

schedmode=1时，JIT在`__global__`入口生成`__schedmode__(1)`限定符（batch模式）。Kernel内使用pypto_pro.language.system.sync_all()且存在多流并发时建议设置为1，以降低死锁风险；参与融合时需同时设置CAP_8_DISABLE_BATCHMODE_CHECK。

### 核数按容器扩展

设置CAP_16_BLOCKDIM_SCALE_UP后，Kernel按融合容器的核数启动，Kernel内需使用pypto_pro.language.get_block_num()动态确定实际核数，不能假设核数等于自身配置值。

## 约束说明

- SIMT kernel暂不支持SK适配，设置sk_cap会在编译阶段报错。
- 图编译侧需开启SuperKernel优化，融合才会实际发生（详见[在torch.compile中开启SuperKernel](#enable-superkernel)一节）。
- 早启的两个接口应在相邻子核中配合使用，详见[早启（Early-Start）交互](#early-start)一节。
- sk_cap必须为pypto_pro.language.SkCap成员或其按位或组合，传入裸整数或非法位会被前端拒绝。
