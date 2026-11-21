# AI CPU Kernel任务失败

## 问题现象描述

AI CPU Kernel调用本身正常返回，但在`torch.npu.synchronize()`（或Stream同步）时抛出RuntimeError（Stream错误）。报错中的device_id、stream_id、task_id与retcode以实际环境为准；关键特征是调用处无异常、同步处失败。

```text
RuntimeError: NPU error: ACL_ERROR_RT_STREAM_TASK_FAILED
Error async task report, device_id:1, stream_id:xxx, task_id:xxx, retcode:0x...
```

## 可能原因

- Kernel状态ABI返回非零：函数体中的早退`return 1`表示KERNEL_STATUS_PARAM_INVALID，AI CPU运行时将任务标记为失败，Host在同步时观察到错误。常见触发条件是Kernel内的参数校验分支命中（如行数、核数超出合法范围）。
- 输入数据内容使Kernel走入失败分支（Device上的shape类数据与Host预期不一致）。

## 处理步骤

1. 核对Kernel函数体中所有`return 1`分支的触发条件，确认Host侧传入的Scalar参数与Tensor内容满足约束。
2. 失败任务会污染发起进程的设备上下文：同一进程内后续Device操作可能持续报错。验证失败路径时使用子进程执行，避免影响主测试进程。
3. 需要定位具体失败分支时，将可疑中间值写入输出Tensor的保留字（调试用），或在子进程中缩小输入规模复现。
