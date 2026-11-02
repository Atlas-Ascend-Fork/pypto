# Softmax Operator Quick Start (SIMD)

<!-- md-trans-meta sourceCommit=099ec62fdd650eea5fa0fd14533d5f0f32da389d translatedAt=2026-10-27T02:40:45.328Z pushedAt=2026-10-27T08:54:42.343Z nl=2 -->

## Task and Objective

This section describes in detail how to implement a simple Softmax operator using the PyPTO Pro SIMD programming paradigm and verify its correctness through test cases. After reading this section, you will learn how to use the PyPTO Pro SIMD APIs to define tiles, allocate on-chip storage, move data, perform reduction computation, and perform element-wise computation.

This example processes a Tensor with a fixed shape of [8, 256] and the FP32 data type. It explicitly defines the tiles used in the computation and their on-chip storage, and performs Softmax on each row along the last dimension through the SIMD API. The first dimension is set to 8 to meet the storage alignment requirement of the FP32 row reduction result tile.

## Operator Design Specification

**Table 1** Softmax operator design specification

| name            | shape    | data type | format |
| --------------- | -------- | --------- | ------ |
| **inputs**  | (8, 256) | float32   | ND     |
| **outputs** | (8, 256) | float32   | ND     |

* Mathematical expression

  $M = \max(z),\quad \text{SoftMax}(z_i) = \frac{\exp(z_i-M)}{\sum_j \exp(z_j-M)}$

* Main APIs used

  Tile definition and allocation APIs: pypto_pro.language.TileType, pypto_pro.language.make_tile_group

  Data movement APIs: pypto_pro.language.load, pypto_pro.language.store

  Basic computation APIs: pypto_pro.language.maximum, pypto_pro.language.expand_sub, pypto_pro.language.exp, pypto_pro.language.sum, pypto_pro.language.expand_div

## Importing PyPTO Pro Modules

Before implementing the Softmax operator, you need to import the PyPTO Pro, PyTorch, and torch_npu modules. The PyPTO Pro module is used to define and compile kernels, while the PyTorch and torch_npu modules are used to construct input data, execute NPU computation, and verify the results.

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu
```

Also define the data shape processed in this example:

```python
ROWS = 8
COLS = 256
```

## Core Code Logic

1. Define the tile types and on-chip storage space.

   In PyPTO Pro SIMD programming, you need to explicitly define the tile types used during computation. In this example, the shape of the input, intermediate result, and output tiles is [8, 256], and the shape of the reduction result tiles for the maximum value and the sum of exponentials is [8, 1].

   ```python
   tile_type = pl.TileType(
       shape=[ROWS, COLS],
       dtype=pl.DT_FP32,
       target_memory=pl.MemorySpace.Vec,
   )

   reduce_type = pl.TileType(
       shape=[ROWS, 1],
       dtype=pl.DT_FP32,
       target_memory=pl.MemorySpace.Vec,
       layout=pl.DN,
   )
   ```

   Use pypto_pro.language.make_tile_group to allocate Vector-side UB space for different computation stages. Different tile groups use different on-chip addresses to avoid overlapping storage space.

   ```python
   src_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x0000,
       mutex_ids=[0],
   )

   tmp_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x2000,
       mutex_ids=[1],
   )

   work_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x4000,
       mutex_ids=[2],
   )

   out_group = pl.make_tile_group(
       type=tile_type,
       addrs=0x6000,
       mutex_ids=[3],
   )

   max_group = pl.make_tile_group(
       type=reduce_type,
       addrs=0x8000,
       mutex_ids=[4],
   )

   sum_group = pl.make_tile_group(
       type=reduce_type,
       addrs=0x8100,
       mutex_ids=[5],
   )
   ```

2. Implement the Softmax kernel function.

   Define the Softmax kernel function using the pypto_pro.language.jit decorator, and enable automatic Tile Mutex management by setting auto_mutex=True.

   The Softmax computation sequentially performs data loading, maximum value reduction, subtraction of the maximum value, exponential computation, sum reduction, normalization, and data unloading.

   ```python
   @pl.jit(auto_mutex=True)
   def softmax_simd_kernel(
       src: pl.Tensor[[ROWS, COLS], pl.DT_FP32],
       dst: pl.Tensor[[ROWS, COLS], pl.DT_FP32],
   ):
       tile_type = pl.TileType(
           shape=[ROWS, COLS],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
       )

       reduce_type = pl.TileType(
           shape=[ROWS, 1],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
           layout=pl.DN,
       )

       src_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x0000,
           mutex_ids=[0],
       )

       tmp_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x2000,
           mutex_ids=[1],
       )

       work_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x4000,
           mutex_ids=[2],
       )

       out_group = pl.make_tile_group(
           type=tile_type,
           addrs=0x6000,
           mutex_ids=[3],
       )

       max_group = pl.make_tile_group(
           type=reduce_type,
           addrs=0x8000,
           mutex_ids=[4],
       )

       sum_group = pl.make_tile_group(
           type=reduce_type,
           addrs=0x8100,
           mutex_ids=[5],
       )

       with pl.section_vector():
           src_tile = src_group.current()
           tmp_tile = tmp_group.current()
           work_tile = work_group.current()
           out_tile = out_group.current()
           max_tile = max_group.current()
           sum_tile = sum_group.current()

           # 1. GM -> UB
           pl.load(src_tile, src, [0, 0])

           # 2. Calculate the maximum value of each row
           pl.maximum(max_tile, src_tile, tmp_tile, dim=0)

           # 3. x - max(x)
           pl.expand_sub(work_tile, src_tile, max_tile, dim=0)

           # 4. exp(x - max(x))
           pl.exp(work_tile, work_tile)

           # 5. sum(exp(x - max(x)))
           pl.sum(sum_tile, work_tile, tmp_tile, dim=0)

           # 6. exp(x - max(x)) / sum(exp(x - max(x)))
           pl.expand_div(out_tile, work_tile, sum_tile, dim=0)

           # 7. UB -> GM
           pl.store(dst, out_tile, [0, 0])
   ```

   Here, pypto_pro.language.maximum and pypto_pro.language.sum perform reduction along the last dimension. For an input of shape [8, 256], the shape of both reduction results is [8, 1].

   pypto_pro.language.expand_sub and pypto_pro.language.expand_div broadcast the reduction result of Shape [8, 1] to [8, 256], completing the subtraction of the maximum value and probability normalization, respectively.

## Test Case

To verify the correctness of the Softmax operator, a PyTorch Tensor is used as the input, the computation is performed through the PyPTO Pro Kernel, and the result is compared with that of the built-in Softmax function in PyTorch. Before executing the PyPTO Pro and PyTorch related code, specify the corresponding Device ID.

```python
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)

torch.manual_seed(0)

src = torch.randn(
    ROWS,
    COLS,
    dtype=torch.float32,
    device=device,
)

dst = torch.empty_like(src)

softmax_simd_kernel(src, dst)

torch.npu.synchronize()

golden = torch.softmax(src, dim=-1)

torch.testing.assert_close(
    dst,
    golden,
    rtol=1e-4,
    atol=1e-5,
)

print("SIMD Softmax kernel passed!")
```

## Compilation and Execution

Run the following command in an environment where PyPTO Pro is installed:

```bash
# Configure the CANN environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the device ID
export TILE_FWK_DEVICE_ID=0

# Run the script
python3 softmax_simd.py
```

After the program is successfully executed, the following information is displayed:

```text
SIMD Softmax kernel passed!
```
