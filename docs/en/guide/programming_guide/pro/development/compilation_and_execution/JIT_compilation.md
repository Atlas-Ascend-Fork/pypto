# JIT Compilation

<!-- md-trans-meta sourceCommit=c5e0cd8968ede57b6becf06017dfd0301c567ef2 translatedAt=2026-11-15T02:23:33.343Z pushedAt=2026-11-18T06:03:01.361Z nl=2 -->

A Kernel declared with `@pypto_pro.language.jit()` completes parsing, code generation, and compilation upon its first launch, without requiring a separate compilation command. JIT is suitable for kernel development, function verification, and performance debugging.

## JIT Compilation Process

When a Kernel is launched for the first time, the following steps are executed in sequence:

1. Bind the Kernel arguments and launch configuration, and determine the dynamic parameters and compile-time specialization information.
2. Parse the Kernel function body, generate PyPTO IR, and validate it.
3. Apply the necessary transformations to the IR, and generate the Device code and the Host-side Launcher.
4. Compile and load the artifacts, and then dispatch the Kernel to the specified Stream.

Both compilation and execution are triggered by a single Kernel call. Kernel launch is asynchronous with respect to the Host, but the first call waits for the current compilation instance to finish generating.

## Triggering JIT Compilation

JIT compilation is triggered when a kernel is launched for the first time through `kernel[stream, block_dim](...)`. The following uses the defined `add_kernel` as an example to show the first launch and the reuse under the same compilation signature:

```python
import torch
import torch_npu


x = torch.rand(64, 64, device="npu:0", dtype=torch.float16)
y = torch.rand_like(x)
out = torch.empty_like(x)

# First launch: compile and execute.
add_kernel[None, 1](x, y, out)
torch.npu.synchronize()

# Reuse the compilation result in the current process when the compilation signature is the same.
add_kernel[None, 1](x, y, out)
```

For the definition and launch syntax of a kernel, see [Kernel Function](../kernel_function.md).

### Querying the Available Number of Cores

Before setting `block_dim`, it is recommended to use the PyTorch NPU API to query the maximum number of cores currently available on the target Stream:

```python
import torch
import torch_npu


stream = torch.npu.current_stream()
limits = torch.npu.get_stream_limit(stream)
max_vector_blocks = limits["vector_core_num"]
max_cube_blocks = limits["cube_core_num"]
max_mixed_blocks = min(max_cube_blocks, max_vector_blocks // 2)
```

Pure Vector, pure Cube, and mixed Kernels with an AIC:AIV ratio of 1:2 use the corresponding `max_*_blocks` as the upper limit of `block_dim`, and then select the number of logical cores based on the number of tasks. If you only need to query the Device-level limit, you can use `torch.npu.get_device_limit(torch.npu.current_device())`; when launching on a specified Stream, the query result of the target Stream prevails. For the meaning and calling form of `block_dim`, see [Kernel function](../kernel_function.md#meaning-and-setting-of-blockdim).

## Compilation Signature and Reuse

The same Kernel object distinguishes compilation instances by compilation signature. The following information may produce different instances:


| Information                           | Impact on Compilation Instance                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Fixed Tensor dimensions                 | Dimensions declared as fixed values are compiled into the IR when the Kernel is defined. The corresponding dimensions of the Tensor must match at call time.                                   |
| pypto_pro.language.STATIC dimensions  | The runtime value participates in specialization, and a new instance is generated when the value changes.                                                                   |
| pypto_pro.language.DYNAMIC dimensions | The dimension value does not participate in specialization, and the instance is reused when the value changes.                                                                       |
| Scalar input parameters (DT_*)             | The data type is determined by the Kernel parameter annotation, and the actual argument value is passed in at runtime; the instance is reused when the value changes.                                       |
| TilingKey                      | Each valid Key corresponds to a dedicated instance.                                                                              |
| datatype                       | Each data type combination corresponds to a dedicated instance.                                                                         |
| Compilation target                       | An explicitly specified target is determined when the Kernel object is created; if not specified, it is determined based on the runtime environment at first launch. Different targets use different Kernel objects. |

For the declaration of static and dynamic shapes, see [Tensor Creation and Operations](../tensor_creation_and_operations.md). Scalar input parameters and TilingData fields are both runtime data, and changes in their values do not independently produce compilation instances. For the difference between TilingData and TilingKey, see [Tiling Parameter Definition and Passing](../tiling/tiling_parameter_definition.md).

`stream` and `block_dim` only affect the current launch and do not participate in the compilation signature; adjusting either of them does not generate a new compilation instance.

The JIT reuse scope is limited to the current Python process. After the process is restarted, the generation and compilation process is re-executed; the files in the `build` directory are used for loading and debugging, and are not used as a cross-process persistent cache.

## JIT Decorator Parameters

`pypto_pro.language.jit` supports the following configurable parameters:


| Parameter | Type | Description | Default |
| ----------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| mode | str or None | Specifies the programming mode of the Kernel. Both SIMD and SIMT programming are supported. When "simd", None, or omitted, SIMD programming is used; when "simt", SIMT programming is used. | None |
| arch | str | Specifies the compilation target. It is usually omitted and automatically determined by the runtime environment. | None |
| auto_mutex | bool | Specifies whether to automatically insert synchronization for data dependencies that the framework can recognize, based on the mutex metadata declared by TileGroup. | True |
| name | str | Custom kernel name, used to distinguish compilation artifacts. If not set, the name of the decorated function is used. | None |
| pipeline | pypto_pro.language.pipeline.PipelineConfig | Configures the automatic CV parallel pipeline transformation. | None |
| sanitizer | bool | Enables kernel memory checking to detect out-of-bounds GM access, out-of-bounds Tile access, overlapping Tile memory ranges, and incorrectly paired mutexes. The check results are written to a report file. For details, see [Memory Checking](../../debug/sanitizer.md). | False |
| tiling_key | TilingKey definition class | Binds the TilingKey fields and their candidate values supported by the current kernel. | None |
| datatype | dict[str, str] | Declares the kernel parameters that require datatype specialization, as well as the datatype variable names referenced in the kernel function body. | None |
| compile_timeout | int | Sets the compilation timeout for the current kernel, in seconds. | None (the base default value is 600 seconds) |
| max_threads | int or None | The maximum number of threads allowed per Thread Block for a SIMT kernel, in the range [1, 2048]. Required in SIMT mode; this parameter is not accepted in SIMD mode. | None |
| sk_cap | pypto_pro.language.SkCap | SuperKernel capability bits. When set, in addition to the `__global__` entry, a `__sk__` entry and an `SK_BIND` record are generated, and the Kernel can be fused by the graph compilation side. Combine `CAP_0_SK_ONLY`, `CAP_1_EARLY_START_WAIT`, `CAP_2_EARLY_START_SET`, `CAP_4_DISABLE_DCCI_ALL`, `CAP_8_DISABLE_BATCHMODE_CHECK`, and `CAP_16_BLOCKDIM_SCALE_UP` with bitwise OR. `None` means no SK entry is generated, and `CAP_0_SK_ONLY` means only the SK binding is generated without enabling other features. Only members of pypto_pro.language.SkCap and their bitwise OR combinations are accepted; raw integers are not accepted. | None |
| schedmode | int | Kernel scheduling mode, written to the `__schedmode__(mode)` qualifier of the `__global__` entry: 0 for normal mode and 1 for batch mode. | 0 |

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True, compile_timeout=1200, name="add_kernel")
def add_kernel(x, y, out):
    ...
```

`auto_mutex=True` only handles data dependencies that the framework can recognize through TileGroup, and cannot replace all explicit synchronization. When TileGroup mutex metadata is not used, or when the dependency relationships cannot be recognized by the framework, explicit synchronization must be performed based on the data flow.

`tiling_key` and `datatype` are used to generate compile-time specialization instances and are not passed as formal parameters of the kernel function. When launching the kernel, you need to provide the corresponding TilingKey and datatype dictionaries in the square brackets. For the definitions and launch parameter positions of the two, see [Kernel Function](../kernel_function.md#using-tilingkey-and-datatype). For TilingKey field rules, see [Tiling Parameter Definition and Passing](../tiling/tiling_parameter_definition.md#tilingkey).

`pipeline` accepts a `pypto_pro.language.pipeline.PipelineConfig` object, which is used to configure the automatic CV parallel pipeline transformation. If it is not set, this transformation is not performed. For configuration methods and usage constraints, see [Automatic CV Parallel Pipeline](../../advanced_programming/auto_parallel_pipeline.md).

`sk_cap` is used for SuperKernel (SK) fusion adaptation. Its value is a member of pypto_pro.language.SkCap or a bitwise OR combination of members. `None` means no SK entry is generated, and `CAP_0_SK_ONLY` means an SK entry is generated without enabling other capability bits. When set, JIT additionally generates the `__sk__` entry and the `SK_BIND` record in the Host-side Launcher, and appends SK-related header file paths and compilation macros during the compilation phase; the content of `kernel.cpp` is the same as in a normal build. The meanings of the capability bits are as follows:


| Member | Bit Value | Description |
| --------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `CAP_0_SK_ONLY` | 0 | Only generates the SK entry and `SK_BIND`, without enabling the early-start, DCCI, batchmode, or core-scaling features |
| `CAP_1_EARLY_START_WAIT` | 1 | Calls pypto_pro.language.system.wait_pre_task_end inside the kernel, allowing instructions before the call point to overlap with the preceding sub-kernel |
| `CAP_2_EARLY_START_SET` | 2 | Calls pypto_pro.language.system.set_next_task_start inside the kernel, allowing instructions after the call point to overlap with the following sub-kernel |
| `CAP_4_DISABLE_DCCI_ALL` | 4 | The framework skips DCCI before and after sub-kernel calls, often used as the default value when only binding is required |
| `CAP_8_DISABLE_BATCHMODE_CHECK` | 8 | Skips the `__schedmode__` check so that a Kernel with `schedmode=1` can still participate in fusion |
| `CAP_16_BLOCKDIM_SCALE_UP` | 16 | Indicates that the operator already supports scaling up the number of cores. After fusion, the SuperKernel launches the operator with its own number of cores; the operator must ensure correct execution when the actual number of launched cores is greater than its original number of cores. |

`sk_cap` only declares that the Kernel can be fused; whether fusion occurs is determined by the graph layer. When not fused, the Kernel executes as a normal Kernel, and the early-start APIs have no effect. The early-start bits must be used together with the APIs in the paired sub-kernels (the preceding sub-kernel uses set, and the following sub-kernel uses wait), and each core and each TilingKey branch calls them only once.

`schedmode` only affects how the Kernel is dispatched: `0` is normal mode and `1` is batch mode, based on which JIT generates the `__schedmode__` qualifier on the entry. When pypto_pro.language.system.sync_all() is used inside the Kernel and multiple streams run concurrently, it is recommended to set it to `1` to reduce the risk of deadlock.

## Kernel Dispatch and Synchronization

After compilation is complete, JIT submits the Kernel to the specified Stream through the Host-side Launcher. In SIMD mode, `kernel[None, block_dim](...)` uses the current Stream, and `kernel[stream, block_dim](...)` uses an explicit Stream. For the launch method of a SIMT kernel, see [Launching a SIMT Kernel from the Host](../vector_computation/simt_computation.md#launching-a-simt-kernel-from-the-host).

Kernel dispatch is an asynchronous operation. Before reading the output, checking accuracy, or measuring the complete execution time, synchronize the corresponding Stream:

```python
import torch
import torch_npu


stream = torch.npu.Stream()
add_kernel[stream, num_cores](x, y, out)
stream.synchronize()
```

For a complete description of Stream and `block_dim`, see [Kernel Function](../kernel_function.md#calling-a-kernel).

## Compilation Artifacts

When `ASCEND_WORK_PATH` is not set, JIT artifacts use `./build/` as the root directory; when it is set, they use `${ASCEND_WORK_PATH}/PYPTO_PRO/build/` as the root directory. The Kernel directory name starts with `{kernel_name}__{arch}`; when STATIC dimension specialization exists, `__shape_{hash}` is appended; device and distributed process identifiers are also appended based on the environment, such as `__d0`, `__r1`, or `__d0_r1`. The device identifier is the current NPU device number, and Rank is the distributed process number, not the number of Tensor dimensions. The datatype and TilingKey instances additionally use the `dt_{hash}` and `tk_{packed}` subdirectories respectively (`tk_none` when TilingKey is not used). The main files include:


| File                  | Description                                           |
| ----------------------- | ------------------------------------------------ |
| kernel.cpp            | Device-side source code generated by CodeGen.                    |
| call_kernel.cpp       | Host-side Launcher source code, responsible for parameter packing and kernel dispatch. |
| call_kernel_{hash}.so | Compiled Launcher shared library.                       |
| *_tiling.h            | Structure header file generated when TilingData is used.           |

When compilation fails, prioritize combining the error log and `kernel.cpp` to locate parsing, code generation, or toolchain issues. After successful compilation, the intermediate source code is retained in the artifact directory by default and can be used to verify the generated code.
