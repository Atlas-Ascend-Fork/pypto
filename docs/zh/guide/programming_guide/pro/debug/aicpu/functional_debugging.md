# 功能调试

AI CPU Kernel的问题分为两类：编译（首次启动）报错和任务执行失败。Kernel调用本身正常返回，任务失败在Stream同步时以错误形式上报；定位时应先区分失败发生在编译阶段还是执行阶段，再沿相应路径排查。

## 调试路径总览

| 现象 | 定位路径 |
|---|---|
| 首次调用时抛出NotSupported、InvalidType等前端异常 | 编译报错：检查Kernel声明与函数体是否超出AI CPU编程面 |
| 首次调用时抛出RuntimeFailure，报错包含AICPU kernel compile failed | 编译报错：检查环境与生成源码 |
| 调用正常返回，`torch.npu.synchronize()`时抛出Stream错误 | 任务执行失败：定位命中的失败分支 |

AI CPU Kernel不支持AI Core侧的调试机制：装饰器选项sanitizer会被拒绝，AI Core Error异常dump不适用于控制CPU任务，[功能调试](../ai_core/functional_debugging.md)中的内存检测和异常复现工具不覆盖AI CPU Kernel。

## 编译报错

### 前端拒绝

函数体或装饰器选项超出AI CPU编程面时，首次启动在前端解析阶段抛出NotSupported，报错信息点名不支持的接口或选项：

- 装饰器选项：仅接受arch、name、compile_timeout、max_threads；pipeline、tiling_key、datatype、sanitizer、sk_cap、schedmode会被拒绝，auto_mutex被强制关闭；
- Tensor操作面：仅支持[pypto_pro.language.numel](../../../../../api/pro_api/AICPU-API/numel.md)、pypto_pro.language.getval、pypto_pro.language.setval与`is None`/`is not None`判断；Tensor方法形式（如`t.size()`）、Tile接口、VF/SIMT函数与同步类系统接口均不支持；
- 参数种类：TILING、WINDOW参数在参数校验阶段拒绝。

按报错hint指明的编程面改写Kernel，编程面与能力边界参见[AI CPU编程范式](../../programming_paradigm/AICPU/programming_paradigm.md)。

### 工具链编译失败

报错信息包含`AICPU kernel compile failed`、`AICPU host compile failed`或`AICPU link failed`，并附带完整的bisheng命令与stderr：

1. 确认`ASCEND_HOME_PATH`指向CANN toolkit安装根目录，且包含`<prefix>-linux`目录和`toolkit/toolchain/hcc`工具链；
2. 核对报错附带的bisheng命令与stderr，区分环境问题与源码问题；
3. 检查产物目录中的`{op}_aicpu.cpp`与`{op}_aicpu_host.cpp`生成源码，确认失败位置，产物说明参见[JIT编译](../../development/compilation_and_execution/JIT_compilation.md#编译产物)。

## 任务执行失败定位

AI CPU Kernel遵循状态ABI：函数体中的裸`return`表示成功（KERNEL_STATUS_OK），`return 1`表示参数非法（KERNEL_STATUS_PARAM_INVALID）。非零状态由AICPU运行时标记任务失败，Host在同步Stream时观察到错误，报错形如`ACL_ERROR_RT_STREAM_TASK_FAILED`。

定位步骤：

1. 核对Kernel函数体中所有`return 1`分支的触发条件，确认Host侧传入的Scalar参数与Tensor内容满足约束；
2. 需要定位具体失败分支时，将可疑中间值写入输出Tensor的保留字（调试用）；
3. 失败任务会污染发起进程的设备上下文，同一进程内后续Device操作可能持续报错：验证失败路径时在子进程中执行，避免影响主测试进程。

常见报错（Stream同步错误、启动形式报错、NPU Tensor校验等）的现象、原因与处理步骤参见[常见问题](../../../../appendix/faq/index.md)中的AI CPU Kernel条目。
