# 功能调试

AI CPU Kernel的问题分为两类：编译（首次启动）报错和任务执行失败。Kernel调用本身正常返回，任务失败会在流同步时报错；定位时应先区分失败发生在编译阶段还是执行阶段，再沿相应路径排查。

## 调试路径总览

| 现象                                                    | 定位路径                                                     |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| 首次调用时抛出NotSupported、InvalidType等前端异常       | 编译报错：检查Kernel声明与函数体是否超出AI CPU支持的接口范围 |
| 调用正常返回，`torch.npu.synchronize()`时抛出流同步失败 | 任务执行失败：定位触发了哪个校验分支                         |

AI CPU Kernel不支持AI Core侧的调试机制：装饰器选项sanitizer会被拒绝，AI Core Error异常dump不适用于控制CPU任务，[功能调试](../ai_core/functional_debugging.md)中的内存检测和异常复现工具不覆盖AI CPU Kernel。

## 编译报错

### 前端校验拦截

函数体或装饰器选项超出AI CPU支持的接口范围时，首次启动在前端解析阶段抛出NotSupported异常，详细范围如下：

- 装饰器选项：仅支持arch、name、compile_timeout、max_threads；不支持pipeline、tiling_key、datatype、sanitizer、sk_cap、schedmode等选项配置；
- auto_mutex：强制关闭，AI CPU Kernel单任务串行执行，没有设备侧mutex加锁机制；
- 支持的Tensor操作：仅支持[pypto_pro.language.numel](../../../../../api/pro_api/AICPU-API/numel.md)、pypto_pro.language.getval、pypto_pro.language.setval与`is None`/`is not None`判断；Tensor对象方法调用（如`t.size()`）、Tile接口、VF/SIMT函数与同步类系统接口均不支持；
- 参数种类：TILING、WINDOW参数在参数校验阶段拒绝。

接口范围与能力边界参见[AI CPU编程概述](../../programming_paradigm/AICPU/programming_paradigm.md)。

## 任务执行失败定位

任务执行失败会抛出相关异常信息，报错形如`ACL_ERROR_RT_STREAM_TASK_FAILED`。

常见报错（Stream同步错误、启动形式报错、NPU Tensor校验等）的现象、原因与处理步骤参见[常见问题](../../../../appendix/faq/index.md)中的AI CPU Kernel条目。
