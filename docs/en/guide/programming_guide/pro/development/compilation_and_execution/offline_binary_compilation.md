# Offline Binary Compilation

<!-- md-trans-meta sourceCommit=c5e0cd8968ede57b6becf06017dfd0301c567ef2 translatedAt=2026-11-15T02:23:25.725Z pushedAt=2026-11-18T06:03:01.357Z nl=2 -->

PyPTO Pro Kernel can be integrated into the offline build process of an operator project to generate AI Core binaries distributed with the operator package. Offline compilation requires preparing the Device-side Kernel, Host-side Tiling, and operator project build configuration.

The main steps are as follows:

1. Prepare the operator definition, InferShape, Host-side Tiling, and invocation API.
2. Implement the PyPTO Pro Kernel in the `op_kernel` directory.
3. Enable the PyPTO Pro Kernel in `op_host/CMakeLists.txt`.
4. Fill in TilingData on the Host side, and set TilingKey, BlockDim, and Workspace.
5. Run the operator project build script to generate the operator installation package.

## Preparing the Operator Project

The directories directly related to PyPTO Pro offline compilation are as follows. The upper-level category directories in the actual project are subject to the operator project:

```text
<operator_project>
├── build.sh
├── <op_class>
│   └── ${op_name}
│       ├── examples
│       │   └── test_aclnn_${op_name}.cpp
│       ├── op_host
│       │   ├── ${op_name}_def.cpp
│       │   ├── ${op_name}_infershape.cpp
│       │   ├── ${op_name}_tiling.cpp
│       │   └── CMakeLists.txt
│       ├── op_kernel
│       │   └── ${op_file}.py
│       └── op_graph
└── CMakeLists.txt
```

| file or directory | Description |
| --- | --- |
| `${op_name}_def.cpp` | Defines the inputs, outputs, attributes, and data types of the operator. |
| `${op_name}_infershape.cpp` | Infers the output shape and data type. |
| `${op_name}_tiling.cpp` | Computes TilingData, TilingKey, BlockDim, and Workspace. |
| `${op_file}.py` | Implements the Device-side Kernel. |
| `test_aclnn_${op_name}.cpp` | Calls and verifies the operator through the aclnn API. |
| `op_graph` | Stores the Graph Infer and operator prototype registration required for graph mode. It is not needed when only aclnn is used. |

## Implementing a Kernel with PyPTO Pro

The Kernel file is located at `op_kernel/${op_file}.py`, where `${op_file}` does not include the `.py` suffix and is consistent with the name in CMake.

A Kernel delivered offline must meet the following requirements:

- Exactly one Kernel declared with `@pypto_pro.language.jit` is defined in each `op_kernel/${op_file}.py`.
- Define a TilingKey and bind it to the Kernel through the `tiling_key` parameter.
- Define TilingData using a Python `dataclass`; the Kernel contains exactly one parameter of this type, and this parameter must be at the end of the parameter list.
- The Kernel function name is consistent with the operator Kernel entry name.
- The parameter names of the business inputs and outputs are consistent with the operator prototype, and the formal parameter order is consistent with the order in which they are delivered from the Host side.
- Workspace is placed after all business inputs and outputs and before TilingData, that is, the parameters end with `workspace, tiling`.
- When the Kernel needs to obtain the data type of an input or output, declare the corresponding parameter through `datatype`.

The following shows the Kernel structure of `add_example`, with the computation process omitted:

```python
from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


@dataclass
class AddExampleTilingData:
    total_length: int
    tile_num: int


class AddExampleTilingKey:
    sch_mode = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(
    tiling_key=AddExampleTilingKey,
    datatype={
        "x": "data_dtype",
        "y": "data_dtype",
        "z": "data_dtype",
    },
)
def add_example(
    x: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    z: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: AddExampleTilingData,
):
    x_tensor = pl.make_tensor(x, [tiling.total_length], [1], dtype=data_dtype)
    y_tensor = pl.make_tensor(y, [tiling.total_length], [1], dtype=data_dtype)
    z_tensor = pl.make_tensor(z, [tiling.total_length], [1], dtype=data_dtype)
    ...
```

The key of the `datatype` dictionary is the Kernel parameter name, and the value is the dtype variable name used in the Kernel. When multiple parameters map to the same variable, their actual data types must be consistent. Declare only the parameters whose data types need to be read inside the Kernel.

For the field rules of TilingData and TilingKey, see [Tiling Parameter Definition and Passing](../tiling/tiling_parameter_definition.md). For the definitions of Kernel parameters and execution domains, see [Kernel Function](../kernel_function.md).

## Recompiling ACLNN Static Kernels and SuperKernels

This process reuses the native static compilation and SuperKernel combined compilation processes. After the graph backend collects the actual ACLNN inputs, it calls `pypto_compile_op()` through the compilation wrapper in the OPP. PyPTO generates the computation implementation and AscendC template entry for each TilingKey, and the native `compile_op()` completes Host Tiling, static entry, SuperKernel binding, linking, and metadata generation.

The deployment environment for recompilation must provide a matching PyPTO Python package, binding, and C++ backend, as well as the following content in the operator package:

- The Python compilation wrapper and importable DSL source code and its dependencies.
- The TilingData and TilingKey generation headers consistent with those used in Host Tiling compilation.
- The OpDef configuration, Host Tiling shared library, and ACLNN call API.

Static mode requires determined input and output shapes, as well as value dependency information that meets the requirements. The compiler converts the Host Tiling result of this compilation into Device constants, and the static external entry omits the Tiling pointer in the native compiler. Ordinary Tensor content and addresses remain runtime parameters. Default optional Tensors are omitted from the external entry and restored to `nullptr` according to the prototype slot when the internal computation implementation is called.

SuperKernel requires the combined process in which the static ordinary entry and sub-kernels are compiled together. The ACLNN path no longer requires the operator to declare `sk_cap`: CAP_1/2 are triggered by `WaitPreTaskEnd`/`SetNextTaskStart` in the source code and `super_kernel_optimize_options={"early_start": 1}`; CAP_4 is handled by the native compilation side after `dcci_disable_on_kernel` matches the sub-kernel symbol. Even if the operator retains `sk_cap`, ACLNN ignores it; do not rely on `sk_cap` to control ACLNN capabilities. JIT mode still declares capabilities through `pypto_pro.language.jit(sk_cap=...)`; the ACLNN automatic path does not use `sk_cap`. The current ACLNN automatic path already supports CAP_1/CAP_2, but still does not support CAP_8/CAP_16; it does not support the combination of dynamic shapes and SuperKernel, dynamic Tensor lists, MC2, output shapes that depend on computation results, or the combination of SIMT and SuperKernel. AIC-only SuperKernel synchronization has not yet been verified separately.

The graph side can use the following configuration to initiate recompilation; disable `super_kernel_optimize` when only static compilation is used:

```python
import npugraph_ex
import torch
import torch_npu

config = npugraph_ex.CompilerConfig()
config.static_kernel_compile = True
config.super_kernel_optimize = True
config.super_kernel_optimize_options = {
    "early_start": 1,
    "dcci_disable_on_kernel": [".*<kernel_symbol_fragment>.*"],
}
# After modifying the operator source code, generation headers, or the PyPTO backend, explicitly disable the old static compilation cache during verification.
config.disable_static_kernel_compile_cache = True
compiled = torch.compile(model, backend=npugraph_ex.get_npu_backend(compiler_config=config))
```

`early_start` takes effect only when the source code actually calls the early-start API; `dcci_disable_on_kernel` is configured as needed and matches the generated sub-kernel symbol name rather than `op_type`. Remove the corresponding entry from the dictionary when it is not needed.

During verification, confirm the static installation package of the target operator, the installation and reselect results, the static Kernel names in the captured graph, and the node changes before and after SuperKernel optimization. Some graph backends continue to run the ordinary binary after a recompile failure; passing accuracy alone or enabling a graph-level switch does not prove that static compilation or fusion has taken effect.

## Configuring CMakeLists.txt

Call `enable_pypto_kernel(<op_file>)` in `op_host/CMakeLists.txt`, and place this call before `add_modules_sources` or `add_modules_sources_with_soc`:

```cmake
enable_pypto_kernel(add_example)

add_modules_sources(
    OPTYPE add_example
    ACLNNTYPE aclnn
)
```

The configuration is the same when using `add_modules_sources_with_soc`:

```cmake
enable_pypto_kernel(add_example)

add_modules_sources_with_soc(
    OPTYPE add_example
    ACLNNTYPE aclnn
)
```

`add_example` must be consistent with the file name of `op_kernel/add_example.py`. During the CMake configuration phase, the Kernel file is loaded to generate the TilingData and TilingKey header files shared by Host and Device; in the subsequent build phase, Kernel instances are generated based on datatype and TilingKey.

## Implementing Host-Side Tiling

The build process generates a C++ Tiling class with the same name based on the Python `dataclass` on the Kernel side. The Host side directly fills in the generated type without declaring the structure again:

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddExampleTilingData *tiling =
        context->GetTilingData<AddExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->total_length = total_length;
    tiling->tile_num = tile_num;

    uint64_t tiling_key = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tiling_key);
    context->SetBlockDim(block_dim);

    size_t *workspace_size = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace_size);
    workspace_size[0] = user_workspace_bytes;
    return ge::GRAPH_SUCCESS;
}
```

Host-side Tiling must ensure the following:

- The field names, types, and order of TilingData are consistent with the definitions on the Kernel side.
- The arguments of `GET_TPL_TILING_KEY(...)` are arranged in the order of the TilingKey field definitions and form a valid combination.
- `context->SetBlockDim()` is consistent with the multi-core splitting method of the Kernel.
- The Workspace size covers the user Workspace used by the Kernel and the system Workspace required by related APIs.

`GET_TPL_TILING_KEY(...)` receives the actual candidate values of each field and generates a 64-bit Key based on the field order and candidate subscripts. `context->SetTilingKey()` receives the packed Key and cannot directly accept the unencoded value of a field.

For the meaning and calculation method of BlockDim, see [Meaning and Setting of blockDim](../kernel_function.md#meaning-and-setting-of-blockdim). When system Workspace is required, query the required size through the corresponding platform API and add it to the user Workspace. Do not hardcode a fixed value.

## Compiling Operator Binaries

Before compilation, configure the CANN and compilation toolchain environments, and ensure that the Python environment used for building can import the `pypto_pro` that matches the current source code. Run the following command in the root directory of the operator project:

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example
```

When compiling multiple operators, separate their names with English commas:

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example,other_op
```

To specify a custom operator package name, add `--vendor_name`:

```bash
bash build.sh --pkg --soc=${soc_version} \
    --vendor_name=${vendor_name} \
    --ops=add_example
```

If `--vendor_name` is not specified, the custom operator package name defaults to `custom`.

After the build is complete, the operator installation package is generated in `build_out` in the project root directory. After installing the operator package, you can invoke it through aclnn. For graph mode, you also need to provide the related deliverables in `op_graph`.
