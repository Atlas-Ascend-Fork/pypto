# AI CPU Kernel启动报错

## 问题现象描述

调用Kernel时抛出InvalidType，报错信息包含`AICPU launch requires NPU tensors`。

```text
pypto_pro._errors.InvalidType: [jit.py:769][PRO_RUNTIME]:ErrCode: F00001! Enum: ExternalError::INVALID_TYPE. arg[0] 'x_0': AICPU launch requires NPU tensors, got a tensor on 'cpu'
```

## 可能原因

AI CPU Kernel直接读写Device内存，实参Tensor位于CPU或其他设备。

## 处理步骤

将Tensor迁移到NPU（`tensor.to("npu")`或创建时指定`device="npu"`）后重试。
