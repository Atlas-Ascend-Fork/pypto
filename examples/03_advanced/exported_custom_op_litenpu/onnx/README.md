# onnx/

可运行的 ONNX 导出样例。每个样例是一个目录 `<name>/`，内含五个文件：`kernel.py`（kernel 函数体）、
`op.py`（推导 + CPU 参考实现 + `ExportedCustomOp(...)` 声明）、`model.py`（`nn.Module` + shape 常量）、
`run_demo.py`（运行入口）与 `export_demo.py`（导出入口）。导入链单向分叉：
`kernel -> op -> model -> {export_demo, run_demo}`；任何文件都不得导入这两个入口，因为它们各自以 `__main__`
运行，一旦被导入就会把算子重复声明一次。
`export_demo.py` 把模型导出为 `.onnx` 自定义算子制品；`tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py` 从该制品构建出一个 `libcust_opapi.so` 并放置到 `framework/onnx/` 下供 ATC/GE 发现。

torch 算子与 ONNX 节点无需手写：在 `op.py` 中以配置形式声明在 `ExportedCustomOp` 上（`torch_op_qualname=` +
`onnx_spec=OnnxSymbolicSpec(...)`），pypto 会在导出期间自动生成。`torch_defn`（`run_demo.py` 使用的 CPU
参考实现）也在那里声明。

## 依赖说明

`export_demo.py` 的导出链路依赖 `onnx`，它是**可选依赖**：仅本 ONNX 导出示例需要，PyPTO 核心与
`run_demo.py` 的 CPU/设备运行路径都不依赖它（`pypto.extensions.torch_custom_op_litenpu` 对 `onnx`
采用函数内懒加载）。首次导出前请先安装：

```bash
pip install onnx
```

未安装时 `export_demo.py` 会报 `ModuleNotFoundError: No module named 'onnx'`。

## 版本要求

本示例依赖 `pypto.extensions.torch_custom_op_litenpu`（含 `ExportedCustomOp`、`common/run.py` 等），该扩展
目前仅在 PyPTO 源码 `master` 上提供，随 CANN 9.2.0 版本发布的 PyPTO 不包含这些模块。运行前请先确认实际
加载的是从本仓库源码编译安装的 PyPTO，而不是随 CANN 安装的 PyPTO：

```bash
python3 -c "import pypto; print(pypto.__file__)"
```

若输出指向 CANN 安装路径（例如 `.../Ascend/.../site-packages/pypto`），说明加载的是随包旧版，请参考
《[编译安装](../../../../docs/zh/install/build_and_install.md)》从本仓库源码编译安装 PyPTO 后再运行。运行中若报
`ModuleNotFoundError: No module named 'pypto.extensions...'` 或
`AttributeError: ... has no attribute 'ExportedCustomOp'`，同属版本不匹配，请按上述方式重装，而不是示例或源码缺陷。

请在本目录（`examples/03_advanced/exported_custom_op_litenpu/onnx/`）下运行：

```bash
python3 add/export_demo.py /tmp/add.onnx        # 仅导出，从不运行
python3 ../../../../tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py /tmp/add.onnx
python3 add/run_demo.py                         # 在 cpu 上运行（各算子的 torch_defn），从不导出
python3 add/run_demo.py --device=npu            # 在设备上运行真实 kernel
python3 add/run_demo.py --soc_version=Kirin9030 # 在环境级 NPU 模拟器上运行
```

`run_demo.py` 接受 `--device={cpu,npu}` 与 `--soc_version=<soc>`；`--soc_version` 选择环境级 NPU 模拟器，可与
`--device=cpu` 搭配或省略 `--device`，但绝不能与 `--device=npu` 同时使用。

## 编写规则

DIRECT 的 `@pypto.frontend.jit` kernel 应在其装饰器上固定
`runtime_options={"run_mode": pypto.RunMode.SIM}`。部署出的 `.so` 制品无论如何都会被强制为 SIM，但前端会在导入时按所在机器决定 run_mode，固定为 SIM 可以避免在 CANN 主机上导入 `kernel.py` 时冒出
"NPU is not available" 的意外报错。

每个单 pypto 算子样例都有唯一的词干 `<compute>[_<attr>...]`：计算在前，随后是把该样例与普通版本区分开来的属性。由样例派生的每个标识符都一致使用该词干：目录名 `<stem>`（不带 `_kernel` 后缀）、kernel 函数 `<stem>_kernel`（工厂形式则为
`create_<stem>_kernel`）、推导函数 `<stem>_infer_shape` / `<stem>_infer_dtype`、CPU 参考实现
`<stem>_torch`，以及 torch 注册键 `pypto::<stem>`。

## op_type = 计算 · torch_op_qualname = 每个算子唯一的注册键

`op_type`（`OnnxSymbolicSpec(op_type=...)` 字符串）命名的是计算，它会成为消费方看到的图节点类型
（`"Add"` -> `PyptoCustomOpAdd`），因此在计算相同的样例之间可以重复。`torch_op_qualname`（如
`"pypto::add"`）是 torch 注册键，它携带特性词干；暴露同一算子的样例之间限定名同样会重复，而 torch 注册表是进程全局的，这正是每个样例必须在各自进程中运行的原因。
