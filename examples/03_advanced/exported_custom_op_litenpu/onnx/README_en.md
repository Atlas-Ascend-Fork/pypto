# ONNX Export Examples

These examples demonstrate declaring PyPTO custom operators, exporting ONNX models, and executing the models.

## Examples

| Example | Purpose | Example-specific usage |
|---|---|---|
| [add/](add/) | Declares, exports, and executes an elementwise-addition custom operator. | — |
| [nested_control_flow/](nested_control_flow/) | Demonstrates discovering addition, subtraction, and multiplication operators in nested ONNX `If` subgraphs. The root graph contains no PyPTO operators. | `run_demo.py` accepts `--branch` with `add`, `sub`, or `mul` to select the execution branch. See [export details](nested_control_flow/export_demo.py). |

“—” means no additional options or restrictions beyond the common running instructions.

## Custom-Op Framework Overview

The four diagrams below show how the framework supports these ONNX examples. They cover operator declaration and ONNX export, library build and setup, Python execution, and ATC/GE compilation and launch preparation.

“Repository tools” refers to `tools/exported_custom_op_litenpu`; “Installed extension” refers to `pypto.extensions.torch_custom_op_litenpu`.

**Framework diagram notation:** solid arrows point from a consumer to a module it uses; dashed arrows follow artifact handoffs.

<details>
<summary><strong>Operator declaration and ONNX export</strong> — click to expand diagram</summary>

`ExportedCustomOp` combines the kernel, inference callbacks, optional typed attributes, and optional CPU reference (`torch_defn`); `torch_op_qualname` configures Torch registration and `onnx_spec=OnnxSymbolicSpec(...)` configures ONNX export. In these examples, importing the operator modules registers their Torch operators; export finalization generates and registers their ONNX symbolics.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 320}}}%%
flowchart TB
    subgraph TOOLS["Repository tools"]
        E["<b>export/onnx_export.py</b><br/>onnx_export_session()<br/>resolve_pypto_export_opset()"]
    end

    subgraph EXT["Installed extension"]
        A["<b>common/exported_custom_op.py</b><br/>ExportedCustomOp"]
        AS["<b>common/attr_spec.py</b><br/>AttrSpec<br/>Typed attribute declarations"]
        KS["<b>common/kernel_snippet.py</b><br/>build_kernel_compile_snippet()"]
        OS["<b>onnx/spec.py</b><br/>OnnxSymbolicSpec"]
        T["<b>common/torch_op.py</b><br/>Torch registration and dispatch"]
        F["<b>common/finalize.py</b><br/>finalize_pending_ops()<br/>Finalize ONNX export registration"]
        O["<b>onnx/export.py</b><br/>ONNX symbolics"]
    end

    D["<b>User-defined operator modules</b><br/>op.py / kernel.py / helper modules<br/>Reference · inference · JIT kernel or factory"]

    A -->|"Use kernel<br/>and callbacks"| D
    A -->|"Generate snippet"| KS
    A -->|"Use attributes"| AS
    A -->|"Use export<br/>configuration"| OS
    A -->|"Register<br/>Torch operator"| T
    A -->|"Queue for<br/>export registration"| F
    O -->|"Register<br/>ONNX synthesizer"| F
    E -->|"Finalize<br/>declarations"| F
    E -->|"Read opset floor"| O
    E -->|"Enter export scope"| T

    %% Invisible layout constraints only; these are not dependencies.
    E ~~~ A

    AS ~~~ OS
    OS ~~~ KS
    KS ~~~ D
    T ~~~ F
```

</details>

<details>
<summary><strong>Custom-op library build and setup</strong> — click to expand diagram</summary>

The build and setup tools process exported custom-op nodes and install the native library together with its kernel Python files.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 320}}}%%
flowchart TB
    subgraph TOOLS["Repository tools"]
        CMD["<b>deploy/build_so_and_setup.py</b><br/>Build, then set up"]
        S["<b>deploy/setup.py</b><br/>setup_onnx_custom_op_so()<br/>Install library and kernel Python files"]
        E["<b>export/onnx_export.py</b><br/>Export postprocessing"]
        B["<b>deploy/build.py</b><br/>build_so_from_model()<br/>CMake compile and link"]
        Q["<b>common/discovery.py</b><br/>find_pypto_nodes()"]
        G["<b>deploy/codegen.py</b><br/>Per-operator C++ generation"]
    end

    subgraph EXT[" "]
        EH["Installed extension"]:::heading
        N["<b>common/node_reader.py</b><br/>iter_pypto_nodes() · metadata readers"]
        W["<b>onnx/node_reader.py</b><br/>Graph traversal and attribute readers"]
        AS["<b>common/attr_spec.py</b><br/>AttrSpec<br/>Typed attribute declarations"]
    end

    CMD -->|"Build"| B
    CMD -->|"Set up"| S
    B -.->|"Library and kernel Python files"| S
    B -->|"Discover custom-op nodes"| Q
    B -->|"Generate C++"| G
    N -->|"Read ONNX nodes and metadata"| W
    Q -->|"Find and check custom nodes"| N
    B -->|"Read operator metadata"| N
    E -->|"Validate custom nodes"| N
    E -->|"Traverse nodes for GE domain rewrite"| W
    G -->|"Revalidate attribute records"| AS

    %% Invisible layout constraints only; these are not dependencies.
    G ~~~ E
    Q ~~~ E
    E ~~~ EH
    S ~~~ EH
    EH ~~~ N
    EH ~~~ AS

    classDef heading fill:transparent,stroke:none,font-weight:bold
```

</details>

<details>
<summary><strong>Python execution: reference, simulator and NPU</strong> — click to expand diagram</summary>

Torch dispatch selects the supplied reference implementation or PyPTO kernel execution under a simulator/NPU context. Python execution does not require ONNX export or building the ONNX custom-op library.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 320}}}%%
flowchart TB
    subgraph TOOLS["Repository tools"]
        R["<b>run/demo_run.py</b><br/>Run helpers"]
    end

    subgraph EXT["Installed extension"]
        T["<b>common/torch_op.py</b><br/>Route Torch custom-op calls"]
        U["<b>common/run.py</b><br/>pypto_run_context() · run_op()"]
        C["<b>common/compile.py</b><br/>CompileEntry"]
    end
    D["<b>User-defined operator modules</b><br/>op.py / kernel.py / helper modules<br/>Reference · inference · JIT kernel or factory"]
    P["<b>PyPTO framework APIs</b><br/>pypto.frontend · pypto.runtime · pypto.DT_*<br/>JIT execution · device runtime · data types"]

    R -->|"Import run-context helper"| U
    T -->|"Run kernel with SIM/NPU context"| U
    U -->|"Prepare kernel / infer outputs"| C
    C -->|"Call inference functions / factory"| D
    U -->|"Use supplied JIT kernel"| D
    T -->|"Call reference without a run context"| D
    U -->|"Execute PyPTO kernel"| P
    C -->|"Use PyPTO dtype constants"| P

    %% Invisible layout constraints only; these are not dependencies.
    R ~~~ T
```

</details>

<details>
<summary><strong>ATC/GE compilation and launch preparation</strong> — click to expand diagram</summary>

ATC/GE invokes the installed library's inference and compilation callbacks to obtain kernel artifacts and launch information.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 320}}}%%
flowchart TB
    GE["<b>ATC / GE</b><br/>Load installed custom-op library"]
    subgraph NATIVE[" "]
        NH["Host C++ code in the installed custom-op library"]:::heading
        H["<b>Generated op_host/*.cpp</b><br/>InferShape() / InferDataType()"]
        V["<b>deploy/vendors/pto_custom_op.cpp</b><br/>Compile() / DeclareLaunchArgs()"]
    end
    K["<b>Installed op_kernel/*.py</b><br/>Captured inference functions<br/>__pypto_compile()"]
    C["<b>common/compile.py</b><br/>CompileEntry.__call__()"]
    J["<b>pypto/frontend/parser/entry.py</b><br/>JitCallableWrapper.build_kernel()"]
    OUT["<b>For GE</b><br/>Compiled kernel and launch description"]:::artifact

    GE -->|"Infer shapes and dtypes"| H
    GE -->|"Compile and prepare launch"| V
    H -->|"Call captured inference functions"| K
    V -->|"Call Python compilation entry"| K
    K -->|"Delegate kernel compilation"| C
    C -->|"Build kernel artifact"| J
    V -.->|"Kernel and launch arguments"| OUT

    %% Invisible layout constraints only.
    GE ~~~ NH
    NH ~~~ H
    NH ~~~ V
    classDef heading fill:transparent,stroke:none,font-weight:bold
    classDef artifact fill:#f5f5f5,stroke:#666,stroke-dasharray:5 4
```

</details>

## Running the Examples

### Environment Setup

First follow the [shared prerequisites in the parent guide](../README_en.md#prerequisites).
The additional ONNX requirements are listed below.

#### Dependencies

The export path in `export_demo.py` depends on `onnx`, an **optional dependency**: ONNX export
requires it. PyPTO core and the CPU/device run path of `run_demo.py` do not
(`pypto.extensions.torch_custom_op_litenpu` imports `onnx` lazily inside functions). Install it before the
first export:

```bash
pip install onnx
```

Without it, `export_demo.py` raises `ModuleNotFoundError: No module named 'onnx'`.

#### ATC/GE Requirements

Building and loading the ONNX custom-op library for ATC/GE requires compatible GE development headers and runtime libraries from CANN.

### Execution

Run each example in a separate Python process to avoid conflicts in Python's module cache and Torch's operator registry.

Run from this directory (`examples/03_advanced/exported_custom_op_litenpu/onnx/`). The commands below use `add/`. For another example, replace `add/` in the script paths with its directory and follow its usage notes in the catalogue.

**ONNX export and library setup**

`export_demo.py` exports an ONNX model. `tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py` builds the custom-op library `libcust_opapi.so` from it and installs the library under `framework/onnx/` and kernel Python files under `op_kernel/` for ATC/GE.

```bash
python3 add/export_demo.py /tmp/add.onnx        # export the ONNX model
python3 ../../../../tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py /tmp/add.onnx
```

**Python execution**

`run_demo.py` executes the model using one of the following modes:

```bash
python3 add/run_demo.py                         # run on cpu (each op's torch_defn), never exports
python3 add/run_demo.py --device=npu            # run the real kernels on a device
python3 add/run_demo.py --soc_version=Kirin9030 # run on the env-level NPU simulator
```

Where supported, `run_demo.py` accepts `--device={cpu,npu}` and `--soc_version=<soc>`. The latter selects the environment-level NPU simulator and requires `--device=cpu` or an omitted `--device`; it cannot be combined with `--device=npu`.

## Operator Names

An operator declaration configures its Torch name and its ONNX type separately:

| Configuration | Where it is used |
|---|---|
| `torch_op_qualname="pypto::add"` | The model calls the operator as `torch.ops.pypto.add(...)`. |
| `OnnxSymbolicSpec(op_type="Add", ...)` | Export produces an ONNX node with type `PyptoCustomOpAdd`. |

Set both fields explicitly. The framework adds the `PyptoCustomOp` prefix to the ONNX type; it does not derive that type from the Torch name.
