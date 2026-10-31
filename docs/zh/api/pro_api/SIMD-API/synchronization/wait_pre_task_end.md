# pypto_pro.language.system.wait_pre_task_end

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

SuperKernel是一种算子二进制融合技术；本接口是SK子核函数内的任务间同步接口，用于开启早启（Early-Start）能力中“等待前序子核”的行为。调用位置之前的指令可以与**前序**子核函数并行执行，调用位置之后恢复子核间的保序。

该接口只在核函数被融合为SK子核时生效：需要在[JIT编译](../../../../guide/programming_guide/pro/development/compilation_and_execution/JIT_compilation.md)的sk_cap参数中打开能力位1（pypto_pro.language.SkCap.CAP_1_EARLY_START_WAIT），并由SK框架/图编译侧开启early_start选项（例如TorchAir图后端的`super_kernel_optimize_options={"early_start": 1}`）。未融合或未开启early_start时，早启配置为0，本接口为空操作。

## 函数原型

```python
pypto_pro.language.system.wait_pre_task_end(
    *,
    force: bool = True,
) -> None
```

## 参数说明

| 参数  | 输入/输出 | 说明                                                                                                                                                                                                 |
| ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| force | 输入      | 可选，bool编译期常量，默认True。True时生成`AscendC::WaitPreTaskEnd<true>()`；False时生成`AscendC::WaitPreTaskEnd()`。在SK子核上下文中，默认（FORCE=false）形式会被编译优化掉，因此默认使用强制形式。 |

## 约束说明

- 只能在SK子核函数中调用；普通（未融合）Kernel中调用不产生效果。
- 需与pypto_pro.language.jit的sk_cap参数配套：设置pypto_pro.language.SkCap.CAP_1_EARLY_START_WAIT（如需同时关闭DCCI，可写pypto_pro.language.SkCap.CAP_1_EARLY_START_WAIT | pypto_pro.language.SkCap.CAP_4_DISABLE_DCCI_ALL）。sk_cap含CAP_1/CAP_2时框架自动追加`-D__ASCENDC_SUPERKERNEL_EARLY_START_V3`。
- 与配对子核中的[pypto_pro.language.system.set_next_task_start](set_next_task_start.md)配合使用：二者作用于同一个子核间边界的两侧，相邻的前序子核调用set_next_task_start（set），当前子核调用本接口（wait）。SK图编译按相邻子核配对生成同步任务，未配对时同步任务退化为空同步，不产生重叠效果。
- 每个核、每个TilingKey分支中必须且只能调用一次，避免在运行时if/for中按不同条件调用；所有可能运行的TilingKey分支都应调用，否则同步指令数量不匹配可能卡住。
- 避免与[pypto_pro.language.system.sync_all](sync_all.md)、[pypto_pro.language.system.set_cross_core](set_cross_core.md)的核间同步事件ID冲突；早启同步内部占用核间同步事件ID。
- force必须是编译期bool常量；传入字符串、整数或运行时标量会被前端拒绝。

## 返回值说明

无。

## 调用示例

调用位置之前的代码可与前序子核并行；需要在依赖前序子核输出的指令之前调用，以恢复保序。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, sk_cap=pl.SkCap.CAP_1_EARLY_START_WAIT)
def early_start_wait_kernel(
    x: pl.Tensor[[1024, 64], pl.DT_FP32],
    out: pl.Tensor[[1024, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x0100)

    with pl.section_vector():
        # 调用前的准备工作可与前序子核并行。
        pl.system.wait_pre_task_end(force=True)
        # 以下指令依赖前序子核的输出。
        for row in pl.range(
            pl.get_block_idx(),
            x.shape[0],
            pl.get_block_num() * pl.get_subblock_num(),
        ):
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])
```
