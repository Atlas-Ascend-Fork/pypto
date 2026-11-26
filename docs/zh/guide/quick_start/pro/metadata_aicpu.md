# Metadata算子快速入门（AI CPU）

## 任务与目标

本节介绍如何使用PyPTO Pro AI CPU编程范式实现一个Metadata算子，并通过测试用例验证结果。您将了解如何声明运行在控制CPU上的Kernel、使用标量接口读写GM中的Tensor、通过返回值上报参数错误，以及异步下发任务。

AI CPU Kernel适合控制流密集、串行依赖强、计算量小的任务（如注意力算子的多核切分表生成）。数据密集计算请使用SIMD或SIMT范式，参见[Softmax算子快速入门（SIMD）](softmax_simd.md)与[Softmax算子快速入门（SIMT）](softmax_simt.md)。

## 算子设计规格

**表1** Metadata算子设计规格

| name                              | shape             | data type | format |
| --------------------------------- | ----------------- | --------- | ------ |
| **inputs（输入）** seq_lens | (batch,) 动态     | int32     | ND     |
| **outputs（输出）** plan    | (2 * batch,) 动态 | int32     | ND     |

* 计算逻辑

  对每个批次b：`plan[2 * b] = sum(seq_lens[0:b])`（行偏移），`plan[2 * b + 1] = seq_lens[b]`（行数）。
* 使用的主要接口

  Tensor声明与标量读写接口：pypto_pro.language.Tensor、pypto_pro.language.DYNAMIC、pypto_pro.language.numel、pypto_pro.language.getval、pypto_pro.language.setval

## 导入PyPTO Pro模块

```python
import pypto_pro.language as pl
import torch
import torch_npu
```

## 核心代码逻辑

1. 使用mode="aicpu_kernel"声明Kernel：输入、输出为动态长度Tensor，函数体按批次顺序逐项读写标量。
2. 参数非法时`return 1`上报KERNEL_STATUS_PARAM_INVALID；正常路径以裸`return`结束（KERNEL_STATUS_OK）。
3. Host侧以`kernel(...)`或`kernel[stream](...)`下发，同步后读取结果。

```python
@pl.jit(mode="aicpu_kernel")
def batch_row_plan(
    seq_lens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    plan: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
):
    n = pl.numel(seq_lens)
    if n <= 0:
        return 1  # KERNEL_STATUS_PARAM_INVALID
    offset = 0
    for b in pl.range(0, n):
        rows = pl.getval(seq_lens, b)
        pl.setval(plan, 2 * b, offset)
        pl.setval(plan, 2 * b + 1, rows)
        offset = offset + rows
    return  # KERNEL_STATUS_OK
```

与AI Core Kernel的差异：

- AI CPU Kernel直接对GM做标量计算，计算结果直接输出到GM中；
- 仅支持arch、name、compile_timeout、max_threads等选项，不支持sk_cap、schedmode等AI Core选项；

## 测试用例

将完整示例代码保存为`metadata_aicpu.py`。使用PyTorch Tensor构造输入，将Kernel输出与期望值对比：

```python
import pypto_pro.language as pl
import torch
import torch_npu


@pl.jit(mode="aicpu_kernel")
def batch_row_plan(
    seq_lens: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Input],
    plan: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32, pl.Output],
):
    n = pl.numel(seq_lens)
    if n <= 0:
        return 1
    offset = 0
    for b in pl.range(0, n):
        rows = pl.getval(seq_lens, b)
        pl.setval(plan, 2 * b, offset)
        pl.setval(plan, 2 * b + 1, rows)
        offset = offset + rows
    return


def main():
    torch.npu.set_device(0)
    seq_lens = torch.tensor([8, 16, 32], dtype=torch.int32, device="npu")
    plan = torch.zeros(6, dtype=torch.int32, device="npu")

    batch_row_plan(seq_lens, plan)  # 或 batch_row_plan[stream](seq_lens, plan)
    torch.npu.synchronize()

    expected = [0, 8, 8, 16, 24, 32]
    assert plan.cpu().tolist() == expected
    print("PASS:", plan.cpu().tolist())


if __name__ == "__main__":
    main()
```

## 编译与执行

在已安装PyPTO Pro的环境中运行：

```bash
# 配置CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备ID
export TILE_FWK_DEVICE_ID=0

# 执行脚本
python3 metadata_aicpu.py
```

首次调用触发JIT编译：生成`{op}_aicpu.cpp`（Kernel TU）与`{op}_aicpu_host.cpp`（Host TU），经bisheng交叉编译产出`.so`，详见[JIT编译](../../programming_guide/pro/development/compilation_and_execution/JIT_compilation.md#编译产物)。

程序执行成功后，显示以下信息：

```text
PASS: [0, 8, 8, 16, 24, 32]
```
