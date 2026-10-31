# SuperKernel Sample

This sample shows the minimal usage of **SuperKernel (SK)** for a PyPTO Pro kernel
written in the **SIMD (Tile)** programming style.

## Overview

SuperKernel is an operator binary fusion technique: at graph compilation time, multiple
compiled sub-kernels are fused into one superkernel, reducing task scheduling and launch
overhead. With `@pypto_pro.language.jit(sk_cap=...)`, PyPTO Pro adds an `__sk__` entry and an `SK_BIND`
record to the generated `call_kernel.cpp`, so the kernel can be fused by SuperKernel.

## Sample Features

- **PyPTO Pro script**: uses `import pypto_pro.language as pl`, distinct from the `pypto` (tensor API) samples.
- **SIMD (Tile) kernel**: uses `pypto_pro.language.load` / `pypto_pro.language.exp` / `pypto_pro.language.store` inside
  `with pypto_pro.language.section_vector():`, and partitions row tiles with `pypto_pro.language.get_block_num()` /
  `pypto_pro.language.get_block_idx()`.
- **SK binding**: `@pypto_pro.language.jit(auto_mutex=True, sk_cap=pypto_pro.language.SkCap.CAP_4_DISABLE_DCCI_ALL)` emits the `__sk__` entry and the
  `SK_BIND` record; the normal `__global__` entry is kept for the non-fused path.
- **Fusion with a torch op**: the kernel is exposed through `torch.library.custom_op` to
  `torch.compile`, and is fused with `+ 1.0` into one superkernel through the
  `npugraph_ex` backend with `super_kernel_optimize` enabled.
- **Precision check**: the fused result is compared with eager `torch.exp(x) + 1`.

## Code Structure

- **`superkernel.py`**:
  - `exp_tile_kernel`: SIMD (Tile) elementwise `exp` kernel with `sk_cap=pypto_pro.language.SkCap.CAP_4_DISABLE_DCCI_ALL`.
  - `_exp`: the PyPTO Pro sub-kernel wrapped as a `torch.library.custom_op`.
  - `ExpAddOne`: PyPTO Pro sub-kernel plus torch `+ 1.0`.
  - `test_superkernel`: warm-up, `torch.compile` (SuperKernel enabled), precision check.

## Running the Sample

### Environment Setup

```bash
# Configure the CANN environment variables
# After installation, configure the environment variables. For the actual path of set_env.sh, refer to the following command.
# The environment variable configuration above takes effect only in the current window. You can write the commands into an environment variable configuration file (for example, .bashrc) as needed.

# Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the NPU device ID
export TILE_FWK_DEVICE_ID=0
```

### Execute the Script

```bash
python3 examples/03_advanced/superkernel/superkernel.py
```

### Expected Output

```text
Input shape: (1024, 512)
Output shape: (1024, 512)
PASS: SuperKernel example passed (sk_cap=CAP_4_DISABLE_DCCI_ALL, SIMD Tile exp + torch add)
```

## Precautions

- This sample requires a CANN / torch_npu version that supports SuperKernel. Fusion is
  unavailable when the `npugraph_ex` backend or `super_kernel_optimize` is not supported.
- The first run triggers JIT compilation and takes noticeably longer than subsequent runs.
- To run a non-SK baseline, drop the `sk_cap` argument and set `super_kernel_optimize` to `False`.
