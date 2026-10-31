# pypto_pro.language.system.set_next_task_start

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

SuperKernel是一种算子二进制融合技术；本接口是SK子核函数内的任务间同步接口，用于开启早启（Early-Start）能力中“与后序子核并行”的行为。调用位置之后的指令可以与**后序**子核函数并行执行，是[pypto_pro.language.system.wait_pre_task_end](wait_pre_task_end.md)的对偶接口。

该接口只在核函数被融合为SK子核时生效：需要在[JIT编译](../../../../guide/programming_guide/pro/development/compilation_and_execution/JIT_compilation.md)的sk_cap参数中打开能力位2（pypto_pro.language.SkCap.CAP_2_EARLY_START_SET），并由SK框架/图编译侧开启early_start选项（例如TorchAir图后端的`super_kernel_optimize_options={"early_start": 1}`）。未融合或未开启early_start时，早启配置为0，本接口为空操作。

## 函数原型

```python
pypto_pro.language.system.set_next_task_start(
    *,
    force: bool = True,
) -> None
```

## 参数说明

| 参数  | 输入/输出 | 说明                                                                                                                                                                                                                                                                                                                     |
| ------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| force | 输入      | 可选，bool编译期常量，默认True。True时生成`AscendC::SetNextTaskStart<PIPE_MTE3, PIPE_FIX, true>()`；False时生成`AscendC::SetNextTaskStart()`。在Ascend 950PR&950DT系列产品上，AIV/AIC的默认pipe分别为PIPE_MTE3/PIPE_FIX，强制形式会显式写出。SK子核上下文中默认（FORCE=false）形式会被编译优化掉，因此默认使用强制形式。 |

## 约束说明

- 只能在SK子核函数中调用；普通（未融合）Kernel中调用不产生效果。
- 需与pypto_pro.language.jit的sk_cap参数配套：设置pypto_pro.language.SkCap.CAP_2_EARLY_START_SET（如需同时关闭DCCI，可写pypto_pro.language.SkCap.CAP_2_EARLY_START_SET | pypto_pro.language.SkCap.CAP_4_DISABLE_DCCI_ALL）。sk_cap含CAP_1/CAP_2时框架自动追加`-D__ASCENDC_SUPERKERNEL_EARLY_START_V3`。
- 与配对子核中的[pypto_pro.language.system.wait_pre_task_end](wait_pre_task_end.md)配合使用：二者作用于同一个子核间边界的两侧，当前子核调用本接口（set），相邻的后序子核调用wait_pre_task_end（wait）。SK图编译按相邻子核配对生成同步任务，未配对时同步任务退化为空同步，不产生重叠效果。
- 应放在算子最后一次数据搬运之后，使其后的指令与后序子核并行。
- 每个核、每个TilingKey分支中必须且只能调用一次，避免在运行时if/for中按不同条件调用；所有可能运行的TilingKey分支都应调用，否则同步指令数量不匹配可能卡住。
- 避免与[pypto_pro.language.system.sync_all](sync_all.md)、[pypto_pro.language.system.set_cross_core](set_cross_core.md)的核间同步事件ID冲突；早启同步内部占用核间同步事件ID。
- force必须是编译期bool常量；传入字符串、整数或运行时标量会被前端拒绝。

## 返回值说明

无。

## 调用示例

本示例在SIMD（Tile）kernel的最后一次数据搬运之后调用pypto_pro.language.system.set_next_task_start，使调用位置之后的指令可以与后序子核并行。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, sk_cap=pl.SkCap.CAP_2_EARLY_START_SET)
def early_start_set_kernel(
    x: pl.Tensor[[1024, 64], pl.DT_FP32],
    out: pl.Tensor[[1024, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000)
    tile_out = pl.make_tile(tt, addr=0x0100)

    with pl.section_vector():
        for row in pl.range(
            pl.get_block_idx(),
            x.shape[0],
            pl.get_block_num() * pl.get_subblock_num(),
        ):
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])

        # 最后一次数据搬运算完之后调用，后续指令可与后序子核并行。
        pl.system.set_next_task_start(force=True)
```
