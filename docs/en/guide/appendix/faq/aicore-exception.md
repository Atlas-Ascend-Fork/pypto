# AI Core Kernel Execution Exception

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-27T01:14:52.387Z pushedAt=2026-10-27T02:41:03.801Z nl=2 -->

## Symptoms

An exception occurs during AI Core kernel execution (hardware trap, execution timeout, or core hang), and the error information is as follows:

```text
ErrCode: F1FFFF! Enum: InternalError::COMMON_INNER_ERROR. aicore exception, device_id: 0, stream_id: 61, task_id: 5929, retcode: 507015, kernelName: PyPTO_fusion_kernel_0_mix_aic
```

## Possible Causes

- The kernel code has out-of-bounds memory access.
- Data dependency edges are lost (the consumer reads before the producer finishes writing).
- Tiling/Shape parameters do not match the kernel.
- Issues in the MACHINE scheduling framework itself.

## Solution

1. Prepare the environment.
    - `pypto` and `torch_npu` can be imported normally (`import pypto; import torch_npu` reports no error).
    - `llvm-symbolizer` is installed (`apt install llvm` or `yum install llvm`).
    - Set the environment variables of the CANN package.
        ```bash
        # Default path installation, using the root user as an example (for a non-root user, replace /usr/local with ${HOME})
        source /usr/local/Ascend/ascend-toolkit/set_env.sh

        # Installation to a specified path
        source ${install_path}/ascend-toolkit/set_env.sh
        ```

2. Run the debug_aicore_error.py script to collect fault information and reproduce the single operator.
    ```bash
    python {pypto installation directory}/lib/scripts/debug_aicore_error.py -p <report_dir> -out <output_dir> [-d <device_id>] [-t <timeout>]
    ```
    | Parameter | Description | Constraint |
    |------|------|------|
    | `-p` | Directory that stores the fault information. It is recommended that you set `export ASCEND_WORK_PATH=./wk` before running the test case, so that `./wk` contains all the required information. | The `log/debug/plog`, `extra-info/data-dump`, and `pypto` subdirectories must be included; otherwise, an error is reported and the script exits. Multiple `.pyptokb` files containing AIC Error are not supported. |
    | `-out` | Output directory of the diagnosis report. | — |
    | `-d` | Device ID. If not specified, it is obtained from the `TILE_FWK_DEVICE_ID` environment variable. | Either `-d` or `TILE_FWK_DEVICE_ID` must be specified, and the corresponding NPU card must be available. |
    | `-t` | Timeout (in seconds) for executing the single-operator reproduction script. The default value is 600. You can increase it appropriately when the test case is large or debug logs are enabled. | — |

    Example:
    ```bash
    python {pypto installation directory}/lib/scripts/debug_aicore_error.py -p ./wk -out ./diagnosis -d 0
    ```

3. Interpreting the results.

    The diagnosis conclusion is output to sections 6 to 8 of `<out>/info_<timestamp>/info.txt`:
    | Conclusion | Description |
    |------|------|
    | Section 6 PASS | The single-operator reproduction shows no AIC Error, which is most likely caused by the input data. |
    | Section 6 FAIL, Section 7 PASS | Intra-core synchronization issue. |
    | Section 6 FAIL, Section 7 FAIL, Section 8 PASS | Sub-func (CCE kernel code) issue. |
    | Section 6 FAIL, Section 7 FAIL, Section 8 FAIL | Machine scheduling framework issue. |

    For detailed running logs, see `<out>/info_<timestamp>/debug_info.txt`. The generated standalone single-operator reproduction script is available at `<out>/info_<timestamp>/test_single_op.py`.


## aicore print

### Description

AiCore Print is used to print tensor data and debug information in AI Core kernels, supporting GM, UB, and L1 memory hierarchies and multiple data types.

### APIs

| API Name | Function | Scenarios |
|---------|------|---------|
| **AICORE_LOGD** | Formatted log printing | Print address, scalar, and prompt dimension information [DEBUG level] |
| **AICORE_LOGI** | Formatted log printing | Print address, scalar, and prompt key information [INFO level] |
| **AICORE_LOGW** | Formatted log printing | Print address, scalar, and prompt information that does not meet expectations [WARN level] |
| **AICORE_LOGE** | Formatted log printing | Print address, scalar, and record exception information [ERROR level] |
| **AiCorePrintShape** | Print shape information | View tensor shape dimensions |
| **AiCorePrintGmTensor** | Print GM tensor | View global memory data |
| **AiCorePrintUbTensor** | Print UB tensor | View unified buffer data (AIV kernel only) |
| **AiCorePrintL1Tensor** | Print L1 tensor | View circular buffer data (AIC kernel only) |
| **AiCorePrintL0CTensor** | Print L0C tensor | View accumulator buffer data (AIC kernel only) |

### Supported Data Types

AiCore Print supports the following data types:

**Floating-point types**:

- **fp32**: `float`
- **fp16**: `half`
- **bf16**: `bfloat16_t`

**Integer types**:

- **int8**: `int8_t`
- **uint8**: `uint8_t`
- **int16**: `int16_t`
- **uint16**: `uint16_t`
- **int32**: `int32_t`
- **uint32**: `uint32_t`
- **int64**: `int64_t`
- **uint64**: `uint64_t`

**FP8 type**:

- **fp8_e4m3**: `float8_e4m3_t`
- **fp8_e5m2**: `float8_e5m2_t`
- **fp8_e8m0**: `float8_e8m0_t`
- **hifloat8**: `hifloat8_t`

### Procedure

### 1. Enabling Trace Logging

Modify the configuration file:

`framework/src/interface/configs/tile_fwk_config.json`

```json
"fixed_output_path": true,
"force_overwrite": false,
```

Modify the header file:

`framework/src/interface/machine/device/tilefwk/aicore_print.h`

```cpp
#define ENABLE_AICORE_PRINT 1
```

### 2. Recompiling and Reinstalling

```bash
rm -rf build_out/ && python build_ci.py && pip install build_out/pypto*whl --force-reinstall --no-deps
```

### 3. Adding Print Code to the Kernel CCE File

**Important process notes:**

<b>When to delete the kernel_aic* directory</b>:

- For the first run or when switching cases, delete the **kernel_aic*** directory.
- For repeated runs of the same use case, keep the **kernel_aic*** directory (retain modifications).

**Step 3.1: Generate kernel CCE files on first run**.

For the first run or when switching cases:

```bash
rm -rf kernel_aic* output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

For repeated runs of the same use case (print code already added):

```bash
rm -rf output/ wk/
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && python xxx.py
```

**Step 3.2: Add print code to the generated CCE files**.

View the generated kernel files:

```bash
ls kernel_aicore/*.cpp
```

Modification steps:
(1) Add `#include "tilefwk/aicore_print.h"` at the beginning of the file.
(2) Add a print call at an appropriate location (a synchronization point after data loading or computation).

Print API call format:

```cpp
AICORE_LOGD(param->ctx, "format string", args...);
AICORE_LOGI(param->ctx, "format string", args...);
AICORE_LOGW(param->ctx, "format string", args...);
AICORE_LOGE(param->ctx, "format string", args...);
AiCorePrintShape(param->ctx, Shape2Dim(dim0, dim1), "name");
AiCorePrintGmTensor(param->ctx, (__gm__ T*)addr, end, begin, "name");
AiCorePrintUbTensor(param->ctx, (__ubuf__ T*)addr, end, begin, "name");
AiCorePrintL1Tensor(param->ctx, (__cbuf__ T*)addr, end, begin, l1_staging, "name");
AiCorePrintL0CTensor(param->ctx, (__cc__ T*)addr, end, begin, l0cShape0, l0cShape1, l0c_staging, "name");
```

**Step 3.3: Configure L1/L0C staging buffer (required only for AiCorePrintL1Tensor/AiCorePrintL0CTensor)**.

```cpp
// L1 staging buffer (allocated from workspace)
__gm__ T* l1_staging = (__gm__ T*)(param->funcData->workspaceAddr);

// L0C staging buffer (allocated from workspace, 32-byte alignment required)
__gm__ T* l0c_staging = (__gm__ T*)(param->funcData->workspaceAddr);
```

**Key notes**:
1. On the first run or when switching to a different test case, delete kernel_aic*. For a repeated run of the same test case, keep the modifications.
2. In capture mode, to view the aicore log output, you must add pypto.runtime._device_synchronize() after replay. Otherwise, the aicore log cannot be output.

### 4. Running the Test and Viewing the Print Results

**Important**: The following commands must be executed **completely in one go** (connected with `&&`). Do not split them into multiple commands:

```bash
export ASCEND_PROCESS_LOG_PATH=./wk && export ASCEND_GLOBAL_LOG_LEVEL=1 && rm -rf output/ wk/ && python xxx.py
grep -rn "\[AICORE\]\[Core" ./wk
```

**Command description**:

1. `export ASCEND_PROCESS_LOG_PATH=./wk`: Sets the log output directory to `./wk`.
2. `export ASCEND_GLOBAL_LOG_LEVEL=1`: Sets the log level to INFO (level 1) and enables log output.
3. `rm -rf output/ wk/`: Clears old logs and build artifacts to avoid interference.
4. `python xxx.py`: Runs the test case to trigger kernel compilation and execution.
5. `grep -rn "\[AICORE\]\[Core" ./wk`: searches for and prints all AiCore Print output (including tensor data and debug information).

### Print Examples for Different Data Types

The following examples demonstrate the print usage for each data type. Insert the print code at an appropriate location (such as the synchronization point after TLoad/TAdd).

### Floating-Point Types

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float*)gmTensor_fp32.GetAddr(), 8, 0, "fp32_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ half*)ubTensor_fp16.GetAddr(), 16, 0, "fp16_ub");

__gm__ bfloat16_t* l1_staging_bf16 = (__gm__ bfloat16_t*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ bfloat16_t*)l1Tensor_bf16.GetAddr(), 16, 0, l1_staging_bf16, "bf16_l1");
```

### Integer Types

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ int8_t*)gmTensor_int8.GetAddr(), 16, 0, "int8_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ uint8_t*)ubTensor_uint8.GetAddr(), 16, 0, "uint8_ub");

AiCorePrintUbTensor(param->ctx, (__ubuf__ int16_t*)ubTensor_int16.GetAddr(), 8, 0, "int16_ub");

AiCorePrintGmTensor(param->ctx, (__gm__ uint16_t*)gmTensor_uint16.GetAddr(), 8, 0, "uint16_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ int32_t*)ubTensor_int32.GetAddr(), 16, 0, "int32_ub");

AiCorePrintGmTensor(param->ctx, (__gm__ uint32_t*)gmTensor_uint32.GetAddr(), 8, 0, "uint32_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ int64_t*)gmTensor_int64.GetAddr(), 8, 0, "int64_gm");

AiCorePrintUbTensor(param->ctx, (__ubuf__ uint64_t*)ubTensor_uint64.GetAddr(), 8, 0, "uint64_ub");
```

### FP8 Types (with Platform Restrictions)

```cpp
AiCorePrintGmTensor(param->ctx, (__gm__ float8_e4m3_t*)gmTensor_fp8e4m3.GetAddr(), 8, 0, "fp8e4m3_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e5m2_t*)gmTensor_fp8e5m2.GetAddr(), 8, 0, "fp8e5m2_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ float8_e8m0_t*)gmTensor_fp8e8m0.GetAddr(), 8, 0, "fp8e8m0_gm");

AiCorePrintGmTensor(param->ctx, (__gm__ hifloat8_t*)gmTensor_hf8.GetAddr(), 8, 0, "hifloat8_gm");
```

### Other APIs

AiCorePrintShape:

```cpp
AiCorePrintShape(param->ctx, Shape2Dim(sym_161_dim_0, sym_161_dim_1), "sym_161");
AiCorePrintShape(param->ctx, Shape3Dim(dim0, dim1, dim2));
AiCorePrintShape(param->ctx, Shape4Dim(dim0, dim1, dim2, dim3), "conv_out");
```

L1 Tensor print example:

```cpp
__gm__ half* l1_staging = (__gm__ half*)(param->funcData->workspaceAddr);
AiCorePrintL1Tensor(param->ctx, (__cbuf__ half*)l1Tensor.GetAddr(), 16, 0, l1_staging, "fp16_l1");
```

L0C Tensor print example (L0C data is printed after being moved to the GM staging buffer via DMA):

```cpp
__gm__ int32_t* l0c_staging = (__gm__ int32_t*)(param->funcData->workspaceAddr);
AiCorePrintL0CTensor(param->ctx, (__cc__ int32_t*)l0cTensor.GetAddr(), 1024, 0, 32, 32, l0c_staging, "int32_l0c");
```

AICORE_LOG:
Wherever AICORE_LOGD can be used, AICORE_LOGI/AICORE_LOGW/AICORE_LOGE can also be used in the same way as AICORE_LOGD.
```cpp
AICORE_LOGD(param->ctx, "GM address=%p", ((__gm__ float*)gmTensor.GetAddr()));
AICORE_LOGD(param->ctx, "Shape=[%ld,%ld]", dim0, dim1);
AICORE_LOGD(param->ctx, "INT8 input loaded");
```

### Notes

1. **L1/L0C staging buffer alignment**: The l1_staging and l0c_staging addresses must be 32-byte aligned. The workspaceAddr meets this requirement by default.

2. **Print quantity control**: PRINT_BUFFER_SIZE is currently 128 KB (defined in `framework/src/interface/machine/device/tilefwk/aicpu_common.h`). If an overflow warning is triggered, increase this value and recompile.

3. **AiCorePrintL1Tensor usage notes**:
   <!-- npu="950" id1 -->
   - Ascend 950PR&950DT products: Not supported
   <!-- end id1 -->
   <!-- npu="A3" id2 -->
   - Atlas A3 products: Supported
   <!-- end id2 -->
   <!-- npu="910b" id3 -->
   - Atlas A2 products: Supported
   <!-- end id3 -->

4. **AiCorePrintUbTensor cannot be used in the AIC (Cube core)**: The scalar processor (SP) of the AIC has no physical path to the UB address space, and thus cannot read scalar data from UB. This restriction is enforced at compile time via `static_assert`. Calling `AiCorePrintUbTensor` in an AIC kernel will trigger a compilation error:

   ```text
   error: static assertion failed: [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
   ```

   To check UB data, perform the check in an AIV (Vector core) kernel, or use `AiCorePrintGmTensor` in the AIC to print data that has been moved to GM.

5. **AICORE_LOGX triggers a runtime error when printing UB data values in the AIC**: When `AICORE_LOGD` uses formats such as `%f` and `%d` to print UB data values in an AIC kernel (for example, `((__ubuf__ float*)addr)[521]`), the compiler generates a scalar load instruction from the UB address space. The AIC SP does not support this operation, triggering MPU error 271:

   ```text
   error from aicore error exception, core id is 0, error code = 271
   errorStr: The MPU address access is invalid
   ```

   Using `%p` to print address values (without reading UB data) is safe. **Correct approach**: Do not directly read UB data values in the AIC kernel. Move the UB print logic to the AIV kernel.

6. **Do not use DMA to move UB data to GM for printing**. The AIC (Cube core) does not have an MTE3 DMA engine (intrinsics such as `copy_ubuf_to_gm` and `copy_ubuf_to_gm_align_v2` do not support the cube target). The `OpCoreType` of `TStoreVec` (`OP_UB_COPY_OUT`) is `AIV`, which is exclusive to the Vector core. Calling these APIs in an AIC kernel causes a compilation error:

   ```text
   error: function type '...' of 'copy_ubuf_to_gm' does not support the given target feature
   ```

### FAQs

### Why Is No Print Output Seen?

Ensure that ENABLE_AICORE_PRINT=1 is set, recompilation and installation are complete, the log dump path is specified, the log level is set to info (1), and the grep search results are as expected.

### L1/L0C Print Alignment WARNING

Ensure that l1_staging/l0c_staging addresses are 32-byte aligned. workspaceAddr itself is already aligned.

### Overflow Warning

Reduce the print quantity or increase PRINT_BUFFER_SIZE and recompile.

### AiCorePrintL1Tensor Cannot Find API Definition

The current platform does not support this (check the `SUPPORT_L1_COPY` macro).

### ld.lld: error: undefined symbol

The `ld.lld: error: undefined symbol` link error occurs during compilation, causing the compilation to fail.

**Cause**: When the `parallel_compile` value is greater than 1, CodeGen compiles multiple subgraphs in parallel. In this mode, symbol dependencies between some compilation units are not correctly handled, resulting in a link failure.

**Solution**: Modify `framework/src/interface/configs/tile_fwk_config.json` by setting `parallel_compile` to `1` (one compilation thread, that is, serial compilation). Note: This configuration item indicates the **number of parallel compilation threads**, not a Boolean switch (`1` indicates a single thread, while `128` and other values indicate multi-threaded parallelism). After the modification, rerun to resolve the issue.

```json
"parallel_compile": 1
```

### Compilation Error Occurs When AiCorePrintUbTensor Is Called in an AIC Kernel

When `AiCorePrintUbTensor` is used in an AIC (Cube core) kernel, the compiler triggers `static_assert`:

```text
error: static assertion failed due to requirement '!std::is_same_v<float, float>':
  [AIC UB Print Error] AiCorePrintUbTensor is not supported on AIC (Cube) kernel.
  AIC Scalar Processor cannot scalar-load from UB address space.
  Please use AiCorePrintUbTensor in AIV (Vector) kernel instead,
  or use AiCorePrintGmTensor to print data that has been moved to GM.
```

**Cause**: The scalar processor (SP) of the AIC (Cube core) has no physical path to the UB address space and thus cannot read scalar data from UB.

**Solution**: Move the `AiCorePrintUbTensor` call to an AIV (Vector core) kernel, or use `AiCorePrintGmTensor` to print data that has been moved to GM.

### 8. Using AICORE_LOGD to Print UB Data Values in AIC Kernel Triggers Error 271

The following code is used in the CCE file of the AIC kernel:

```cpp
AICORE_LOGD(param->ctx, "ubTensor val=%f", ((__ubuf__ float*)ubTensor.GetAddr())[521]);
```

An aicore error is triggered at runtime:

```text
error from aicore error exception, core id is 0, error code = 271
errorStr: The MPU address access is invalid
```

**Cause**: `((__ubuf__ float*)addr)[521]` generates a scalar load instruction from the UB address space, which is not supported by the AIC SP. **Note**: Scalar reads from the UB address space in the AIC kernel cannot be trapped by `static_assert` at compile time (because the `__ubuf__` attribute is no longer retained after the variadic template of `AICORE_LOGD` evaluates the parameter expression), nor can they be caught at runtime (such MPU errors are hardware traps with no software recovery mechanism).

**Solution**:

- Move the UB data printing logic to the AIV kernel.
- In the AIC kernel, use only `%p` to print the UB address value (without reading data). This is safe.
- Check the CCE code of the AIC kernel and delete all expressions that perform `[]` subscript access on the UB address space.

### 9. Compilation Error When TStoreVec/copy_ubuf_to_gm Is Used in AIC Kernel to Move Data

Compilation fails when APIs such as `TStoreVec`, `copy_ubuf_to_gm`, and `copy_ubuf_to_gm_align_v2` are called in an AIC kernel:

```text
error: function type 'void (__gm__ void *, __ubuf__ void *, ...)' of 'copy_ubuf_to_gm' does not support the given target feature
```

**Cause**: The Cube core does not have an MTE3 DMA output engine, and all intrinsics that transfer data from the UB source address do not support the cube target. The `OpCoreType` of `TStoreVec` (`OP_UB_COPY_OUT`) is `AIV`, which is a Vector core-specific operation.

**Solution**: This operation can only be used in AIV (Vector core) kernels. Do not perform this operation in AIC kernels. To print UB data, perform the operation within the AIV core.
