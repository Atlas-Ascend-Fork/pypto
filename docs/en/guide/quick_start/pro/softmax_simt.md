# Softmax Operator Quick Start (SIMT)

<!-- md-trans-meta sourceCommit=49c9842f2975007e96468639469c30745f0bd315 translatedAt=2026-10-27T02:41:31.987Z pushedAt=2026-10-28T09:00:50.807Z nl=2 -->

## Task and Objective

This section describes how to implement Softmax using the PyPTO Pro SIMT programming paradigm by referring to the single-Kernel flow of the SIMD version, and verify the results through test cases. You will learn how to load data into on-chip Tiles, perform reduction and element-wise computation within the SIMT function, and store the results back to global memory.

This example processes a Tensor with a fixed shape of [1, 256] and the FP32 data type. One Kernel launches 256 SIMT threads, with each thread responsible for one element in the Tile. The example requires that the input elements be finite values.

## Operator Design Specification

**Table 1** Softmax operator design specification

| name            | shape    | data type | format |
| --------------- | -------- | --------- | ------ |
| **inputs**  | (1, 256) | float32   | ND     |
| **outputs** | (1, 256) | float32   | ND     |

* Mathematical expression

  $M = \max(z),\quad \text{SoftMax}(z_i) = \frac{\exp(z_i-M)}{\sum_j \exp(z_j-M)}$

* Main APIs used

  Tile definition and data transfer APIs: pypto_pro.language.TileType, pypto_pro.language.make_tile_group, pypto_pro.language.load, pypto_pro.language.store

  Thread index and intra-block synchronization APIs: pypto_pro.language.simt.linear_thread_idx, pypto_pro.language.simt.syncthreads

  Exponential and atomic reduction APIs: pypto_pro.language.simt.exp, pypto_pro.language.simt.atomic_max, pypto_pro.language.simt.atomic_add

## Importing PyPTO Pro Modules

Before implementing the Softmax operator, you need to import the PyPTO Pro, PyTorch, and torch_npu modules. The PyPTO Pro module is used to define SIMT functions and compile kernels, while the PyTorch and torch_npu modules are used to construct input data, execute NPU computation, and verify the results.

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu
```

In this example, the length of the last dimension of the input is 256, so the number of SIMT threads is set to 256:

```python
THREADS = 256
```

## Core Code Logic

1. Implement the SIMT computation function on the Tile.

   Consistent with the SIMD version, which first loads data into the Tile and then performs computation, a single SIMT function here completes maximum reduction, exponential summation, and normalization. The 256 threads each process one element, and the stats_tile within the same block is used to store the maximum value and the exponential sum.

   ```python
   @pl.vector_function(mode="simt", max_threads=THREADS)
   def softmax_tile(src_tile, out_tile, stats_tile):
       tid = pl.simt.linear_thread_idx()

       if tid == 0:
           stats_tile[0, 0] = src_tile[0, 0]
           stats_tile[0, 1] = 0.0
       pl.simt.syncthreads()

       value = src_tile[0, tid]
       pl.simt.atomic_max(stats_tile[0, 0], value)
       pl.simt.syncthreads()

       exp_value = pl.simt.exp(value - stats_tile[0, 0])
       pl.simt.atomic_add(stats_tile[0, 1], exp_value)
       pl.simt.syncthreads()

       out_tile[0, tid] = exp_value / stats_tile[0, 1]
   ```

   The first thread initializes the shared reduction state; after each reduction, all threads wait for the results to become visible via syncthreads(). All three barriers are unconditionally reached by all threads. The maximum value is initialized from the first element of the input to avoid an additional FP32 minimum-value constant.

2. Implement the Softmax kernel function.

   As in the SIMD version, the Kernel first allocates on-chip Tile Groups and then performs data load, computation, and data store in the Vector section. The input and output Tiles each occupy 1024 bytes; the reduction-state Tile occupies 32 bytes, with the first two FP32 elements storing the maximum value and the exponential sum, respectively. The three Tile Groups use non-overlapping on-chip addresses and different Mutex IDs.

   ```python
   @pl.jit(arch="3510", auto_mutex=True)
   def softmax_simt_kernel(
       src: pl.Tensor[[1, THREADS], pl.DT_FP32],
       dst: pl.Tensor[[1, THREADS], pl.DT_FP32],
   ):
       tile_type = pl.TileType(
           shape=[1, THREADS],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
       )
       stats_type = pl.TileType(
           shape=[1, 8],
           dtype=pl.DT_FP32,
           target_memory=pl.MemorySpace.Vec,
       )
       src_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
       out_group = pl.make_tile_group(type=tile_type, addrs=0x0400, mutex_ids=[1])
       stats_group = pl.make_tile_group(type=stats_type, addrs=0x0800, mutex_ids=[2])

       with pl.section_vector():
           src_tile = src_group.current()
           out_tile = out_group.current()
           stats_tile = stats_group.current()

           pl.load(src_tile, src, [0, 0])
           softmax_tile[THREADS](src_tile, out_tile, stats_tile)
           pl.store(dst, out_tile, [0, 0])
   ```

   auto_mutex=True, combined with the Mutex IDs of the Tile Groups, automatically handles the pipeline dependencies among data load, SIMT computation, and data store, so there is no need to manually call sync_src/sync_dst. The syncthreads() inside the SIMT function still handles data dependencies between threads and cannot be omitted.

## Test Case

Use PyTorch Tensors to construct a normal input and inputs with large positive and negative offsets, and compare the result of a single Kernel invocation with PyTorch Softmax. Since this example processes only one row, kernel[None, 1] is explicitly used to launch one core; softmax_tile[THREADS] launches 256 SIMT threads in a single AIV.

```python
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"

torch.npu.set_device(device)

torch.manual_seed(0)
for offset in (0.0, 100.0, -100.0):
    src = torch.randn(1, THREADS, dtype=torch.float32, device=device) + offset
    dst = torch.empty_like(src)

    softmax_simt_kernel[None, 1](src, dst)
    torch.npu.synchronize()

    golden = torch.softmax(src, dim=-1)
    torch.testing.assert_close(dst, golden, rtol=1e-4, atol=1e-5)

print("SIMT Softmax kernel passed!")
```

The reduction state is initialized by thread 0 on each SIMT function call, so no intermediate Tensor on the Host side is required. The accumulation order of FP32 atomic addition is not fixed, so verification uses a tolerance rather than bitwise comparison.

## Compilation and Execution

Run the following command in an environment where PyPTO Pro is installed:

```bash
# Configure the CANN environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the device ID
export TILE_FWK_DEVICE_ID=0

# Execute the script
python3 softmax_simt.py
```

After the program is successfully executed, the following information is displayed:

```text
SIMT Softmax kernel passed!
```
