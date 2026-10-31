# SuperKernel 样例 (SuperKernel Sample)

本示例展示 PyPTO Pro 在 **SIMD（Tile）** 编程方式下使用 **SuperKernel（SK）** 的最小用法。

## 总览介绍

SuperKernel 是一种算子二进制融合技术：图编译阶段把多个已编译的子核函数融合成一个超级核函数，减少任务调度与下发开销。PyPTO Pro 通过 `@pypto_pro.language.jit(sk_cap=...)` 在生成的 `call_kernel.cpp` 中增加 `__sk__` 入口和 `SK_BIND` 记录，使 kernel 可以被 SuperKernel 融合。

## 样例代码特性

- **PyPTO Pro 脚本**：使用 `import pypto_pro.language as pl`，与 `pypto`（tensor API）示例区分。
- **SIMD（Tile）核函数**：在 `with pypto_pro.language.section_vector():` 中使用 `pypto_pro.language.load` / `pypto_pro.language.exp` / `pypto_pro.language.store`，并通过 `pypto_pro.language.get_block_num()` / `pypto_pro.language.get_block_idx()` 按核切分行块。
- **SK 绑定**：`@pypto_pro.language.jit(auto_mutex=True, sk_cap=pypto_pro.language.SkCap.CAP_4_DISABLE_DCCI_ALL)` 生成 `__sk__` 入口与 `SK_BIND` 记录；普通 `__global__` 入口保留用于非融合场景。
- **与 torch 算子融合**：kernel 通过 `torch.library.custom_op` 暴露给 `torch.compile`，与 `+ 1.0` 一起经 `npugraph_ex` 后端、开启 `super_kernel_optimize` 后融合为一个超级核。
- **精度校验**：融合后的结果与 eager `torch.exp(x) + 1` 对比。

## 代码结构

- **`superkernel.py`**：
  - `exp_tile_kernel`：SIMD（Tile）逐元素 `exp` kernel，`sk_cap=pypto_pro.language.SkCap.CAP_4_DISABLE_DCCI_ALL`。
  - `_exp`：包装为 `torch.library.custom_op` 的 PyPTO Pro 子核。
  - `ExpAddOne`：PyPTO Pro 子核 + torch `+ 1.0` 的执行模型。
  - `test_superkernel`：预热、`torch.compile`（SuperKernel 开启）、精度校验。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
# 安装完成后请配置环境变量，请用户根据set_env.sh的实际路径执行如下命令。
# 上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
python3 examples/03_advanced/superkernel/superkernel.py
```

### 预期输出

```text
Input shape: (1024, 512)
Output shape: (1024, 512)
PASS: SuperKernel example passed (sk_cap=CAP_4_DISABLE_DCCI_ALL, SIMD Tile exp + torch add)
```

## 注意事项

- 本示例依赖支持 SuperKernel 的 CANN / torch_npu 版本；`npugraph_ex` 后端或 `super_kernel_optimize` 不可用时无法完成融合。
- 首次运行会触发 JIT 编译，耗时明显长于后续执行。
- 若需关闭 SK 做对照，可将 `sk_cap` 参数去掉，并把 `super_kernel_optimize` 置为 `False`。
