# AI CPU Kernel解析时报错：Tensor/Tile操作不支持

## 问题现象描述

编译（首次启动）阶段抛出NotSupported，如`pl.xxx() on a Tensor is not supported in aicpu kernels`、`pl.xxx() on a Tile is not supported in aicpu kernels`或`Unsupported operation call`。报错附带hint指明AI CPU的Tensor操作面：`The aicpu tensor surface is: tensor is None / is not None, pl.numel, pl.getval, pl.setval`。

```text
pypto_pro._errors.NotSupported: [_call_parser.py:767][PRO_PARSER]:ErrCode: F00005! Enum: ExternalError::NOT_IMPLEMENTED_ERROR. Unsupported operation call: x.size()
```

```text
pypto_pro._errors.NotSupported: [_call_parser.py:802][PRO_PARSER]:ErrCode: F00005! Enum: ExternalError::NOT_IMPLEMENTED_ERROR. pl.abs() on a Tensor is not supported in aicpu kernels
```

## 可能原因

AI CPU Kernel的Tensor操作面仅限pypto_pro.language.numel、pypto_pro.language.getval、pypto_pro.language.setval与`is None`/`is not None`判断；Tensor方法形式（如`t.size()`）、Tile接口、VF/SIMT函数与同步类系统接口均不支持。

## 处理步骤

1. 按AI CPU编程面改写函数体（操作面仅[numel](../../../api/pro_api/AICPU-API/numel.md)、[getval](../../../api/pro_api/SIMD-API/transpose_and_element_access/getval.md)、[setval](../../../api/pro_api/SIMD-API/transpose_and_element_access/setval.md)与在位性判断）：Tensor方法形式改为函数调用形式，Tile计算改为标量循环。
2. 数据密集计算拆分到AI Core Kernel，AI CPU Kernel只保留规划与元数据逻辑。
