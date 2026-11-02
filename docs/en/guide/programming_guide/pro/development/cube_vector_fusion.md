# CV Fusion Computation

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T02:02:28.640Z pushedAt=2026-10-27T10:51:47.813Z nl=2 -->

PyPTO Pro supports combining Cube computation and Vector computation in the same Kernel. This document describes the execution model of hybrid Kernels, cross-core data transfer, and synchronization methods, and provides a basic Matmul+Softmax example followed by an advanced FlashAttention example that uses VF and prefetch pipelining.

## Execution Model

When a Kernel contains both [`pypto_pro.language.section_cube()`](../../../../api/pro_api/SIMD-API/controlflow/section_cube.md) and [`pypto_pro.language.section_vector()`](../../../../api/pro_api/SIMD-API/controlflow/section_vector.md), PyPTO Pro executes it as a hybrid Kernel. One logical Block corresponds to one AIC and two AIVs. The AIC performs Cube computation, while the two AIVs execute the same Vector-domain code and distinguish the data each is responsible for through [`pypto_pro.language.get_subblock_idx()`](../../../../api/pro_api/SIMD-API/system_variables/get_subblock_idx.md).

| Execution Unit | Primary Computation | On-chip Storage | Task Index |
|:---|:---|:---|:---|
| AIC | Cube multiply-accumulate | L1 Buffer, L0A Buffer, L0B Buffer, L0C Buffer | [`pypto_pro.language.get_block_idx()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_idx.md) |
| AIV0/AIV1 | Element-wise, reduction, data rearrangement, and Reg computation | UB, Vector Register | `pypto_pro.language.get_block_idx()` and `pypto_pro.language.get_subblock_idx()` |

In a hybrid Kernel, `block_dim` is used to configure the number of AIC/AIV execution groups. The number of AICs is the actual value returned by [`pypto_pro.language.get_block_num()`](../../../../api/pro_api/SIMD-API/system_variables/get_block_num.md), while the number of AIVs must additionally be multiplied by [`pypto_pro.language.get_subblock_num()`](../../../../api/pro_api/SIMD-API/system_variables/get_subblock_num.md). In multi-execution-group scenarios, the Cube side and the Vector side must use a consistent task mapping to prevent different execution groups from reading and writing the same intermediate result. For details about the meaning of `block_dim`, see [Kernel Function](kernel_function.md#meaning-and-setting-of-blockdim); for execution group mapping, see [SIMD Programming Paradigm](../programming_paradigm/SIMD/programming_paradigm.md#pure-vector-pure-cube-and-hybrid-kernels); and for inter-core partitioning, see [Multi-core Tiling](tiling/multi_core_tiling.md).

## Cross-Core Intermediate Data Transfer

AIC and AIV can transfer intermediate results through on-chip buffer or the workspace in GM. The two approaches differ in data path and synchronization requirements, and should be selected based on the shape, dtype, layout, capacity, and usage pattern of the intermediate results.

### Passing Through On-Chip Buffer

The on-chip path does not require writing intermediate results back to GM, and supports the following transfer directions:

- Cube → Vector: Use [`pypto_pro.language.move`](../../../../api/pro_api/SIMD-API/memory_data_movement/move.md) to move the results from the L0C Buffer to the UB. When one AIC distributes data to two AIVs, you can use [`pypto_pro.language.AccToVecMode`](../../../../api/pro_api/SIMD-API/basic_data_structures/AccToVecMode.md) to split along the M axis or N axis.
- Vector → Cube: First obtain the ND/NZ layout required by the Cube in the UB, and then write to the L1 Buffer through [`pypto_pro.language.move`](../../../../api/pro_api/SIMD-API/memory_data_movement/move.md) or [`pypto_pro.language.insert`](../../../../api/pro_api/SIMD-API/memory_data_movement/insert.md).

The producer and consumer must use consistent intermediate Tile shape, dtype, and layout. When one AIC distributes data to two AIVs, it is also necessary to determine whether to split along the M axis or N axis, and to use the rotating Buffers in the same order.

### Passing Through GM

The intermediate result can also be written to the workspace in GM first, and then moved in by the consumer:

```text
Cube: L0C Buffer ──► workspace (GM) ──► UB: Vector
Vector: UB ──► workspace (GM) ──► L1 Buffer: Cube
```

The workspace can store the complete or tiled intermediate tensor, allowing the producer and consumer to use different tiling orders and supporting repeated reads, but at the cost of increased GM read/write traffic. Concurrent logical blocks, pipelined iterations, or AIV subblocks should use non-overlapping regions. If the same region is reused, the producer must wait until the previous round of reads is complete before overwriting it.

## Establishing Cross-Core Synchronization

A synchronization signal only expresses an execution dependency and does not carry data. Regardless of whether the intermediate result resides in an on-chip buffer or in the workspace, as long as one core produces it and another core consumes it, the consumer must be guaranteed to read it only after the data is ready; when a Buffer is to be cyclically reused, the producer must also be guaranteed not to overwrite it until the consumer has finished using it. The order of `pypto_pro.language.section_cube()` and `pypto_pro.language.section_vector()` in the source code cannot by itself substitute for cross-core synchronization, and [`pypto_pro.language.jit(auto_mutex=True)`](compilation_and_execution/JIT_compilation.md#jit-decorator-parameters) is only responsible for Tile-related intra-core pipeline synchronization; when automatic CV pipelining is not enabled, cross-core dependencies must still be described explicitly.

### Explicit Synchronization

[`pypto_pro.language.system.set_cross_core`](../../../../api/pro_api/SIMD-API/synchronization/set_cross_core.md) sends an event after the preceding operations of the producer pipeline complete, and [`pypto_pro.language.system.wait_cross_core`](../../../../api/pro_api/SIMD-API/synchronization/wait_cross_core.md) blocks the subsequent operations of the consumer pipeline. `pipe` indicates the pipeline that must complete before sending and the pipeline that must be blocked while waiting, respectively; the two ends are not required to be the same.

The default [`pypto_pro.language.CrossCoreSyncMode.INTRA_BLOCK`](../../../../api/pro_api/SIMD-API/basic_data_structures/CrossCoreSyncMode.md) mode is used for AIC/AIV synchronization within the same logical block:

- AIC → AIV: AIC sends a data-ready event, and AIV0 and AIV1 wait separately.
- AIV → AIC: AIV0 and AIV1 send completion events separately, and AIC waits for both events.

When synchronization with only one AIV is required, `pypto_pro.language.CrossCoreSyncMode.UNICAST_BLOCK` can be used; `pypto_pro.language.CrossCoreSyncMode.INTER_SUBBLOCK` synchronizes only AIV0/AIV1 within the same logical block; `pypto_pro.language.CrossCoreSyncMode.INTER_BLOCK` synchronizes cores of the same type across multiple logical blocks and does not establish a dependency between AIC and AIV. The synchronization mode must be selected based on the actual producers and consumers, and cannot be determined solely by whether the synchronization is cross-core.

For example, after Cube finishes writing to the workspace, it sends an event from the FIX pipeline; Vector waits for this event in the MTE2 pipeline before reading the workspace:

```python
import pypto_pro.language as pl


with pl.section_cube():
    pl.store(workspace, cube_result, offsets)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

with pl.section_vector():
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
    pl.load(vector_input, workspace, offsets)
```

Events must appear in pairs, and all participating cores must be able to reach the synchronization point in all runtime branches; otherwise, a deadlock may occur. Before the same event ID is reused for another set of dependencies, all SET/WAIT operations of the previous set must be fully consumed. When a buffer is reused in a loop, setting only the forward "data ready" synchronization is insufficient; a reverse "data consumed" synchronization is also required, or different buffer slots must be used for adjacent iterations.

### Automatic Synchronization and Parallel Pipelining

For the configuration and usage of automatic synchronization and parallel pipelining, see [Automatic CV Parallel Pipelining](../advanced_programming/auto_parallel_pipeline.md). Using automatic pipelining requires satisfying the usage constraints described in that document. When these constraints cannot be met, the pipeline should be orchestrated manually, and the `set_cross_core` and `wait_cross_core` APIs described above should be used to explicitly configure cross-core synchronization.

## Basic Example: Matmul+Softmax

The following implements $\operatorname{Softmax}(A \times B)$. The cube multiplication $S=A\times B$ is executed on the Cube side, and the row-wise Softmax is executed on the Vector side:

$$
out_{i,j} = \frac{e^{S_{i,j} - m_i}}{\sum_{j} e^{S_{i,j} - m_i}}, \qquad m_i = \max_{j} S_{i,j}
$$

In this implementation, $S$ is stored in the workspace. Each logical Block is assigned a non-overlapping row block along the M axis: the Cube first writes the $S$ region it is responsible for, and then the two corresponding AIVs read these rows and complete the Softmax. The example calls the Kernel directly and uses the number of available execution groups by default for the current Stream.

- The shape of $A$ is $[M,128]$, and the shape of $B$ is $[128,N]$, where M/N are runtime dynamic dimensions; K is fixed at 128. Logical Blocks are assigned row blocks along the M axis in units of 64 rows, strided by the actual number of execution groups returned by `get_block_num()`.
- The shape of the Cube output Tile is $[64,64]$. Therefore, each M row block is computed once every 64 columns along the N axis, and the result is written to the corresponding region of the workspace with shape $[M,N]$.
- Softmax requires obtaining the maximum value and the sum of exponentials of each row along the full N axis. The Vector cannot load the full N axis into the UB at once, so it traverses in units of 64 columns three times: the first pass merges the row maximum values, the second pass accumulates the sum of exponentials, and the third pass normalizes and writes back.
- A logical Block contains two AIVs. The two AIVs split the M row block of their group, each processing at most 32 rows; the M/N tail blocks are passed to both the Cube and Vector sides through `valid_shape`.
- After the Cube of each group finishes writing the workspace region it is responsible for, it sends an event once, and the Vector of the same group waits for this event before reading. The Cube and Vector within the same group do not overlap execution across N blocks.

### Tile API Implementation

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 64
TILE_N = 64
VEC_ROWS = 32
K_SIZE = 128

@pl.jit(auto_mutex=True)
def matmul_softmax_kernel(a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
                          b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
                          out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
                          workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    M = a.shape[0]
    N = b.shape[1]
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    M_TILES = (M + TILE_M - 1) // TILE_M
    N_TILES = (N + TILE_N - 1) // TILE_N

    # ---- Cube Tile: compute S = A @ B ----
    tt_a_mat = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_b_mat = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left,
                          compact=1)
    tt_right = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Right,
                           compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         compact=1)

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile: each AIV processes up to 32 M rows and completes Softmax in three passes along the N blocks ----
    tt_vec = pl.TileType(shape=[VEC_ROWS, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec)
    tt_red = pl.TileType(shape=[VEC_ROWS, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    # Row reduction uses a DN view of [M half block, 1], and element-wise merging uses the row-major view of the same memory.
    tt_red_rm = pl.TileType(shape=[1, VEC_ROWS], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec)

    qk_vec = pl.make_tile_group(type=tt_vec, addrs=0x0000, mutex_ids=[5])
    tmp_vec = pl.make_tile_group(type=tt_vec, addrs=0x2000, mutex_ids=[6])
    exp_vec = pl.make_tile_group(type=tt_vec, addrs=0x4000, mutex_ids=[7])
    red_vec = pl.make_tile_group(type=tt_red, addrs=0x6000, mutex_ids=[8])
    red_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6000, mutex_ids=[9])
    global_max = pl.make_tile_group(type=tt_red, addrs=0x6100, mutex_ids=[10])
    global_max_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6100, mutex_ids=[11])
    global_sum = pl.make_tile_group(type=tt_red, addrs=0x6200, mutex_ids=[12])
    global_sum_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6200, mutex_ids=[13])

    # ==== Cube: split cores along the M axis, split blocks along the N axis, and write the result to workspace ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_base = mi * TILE_M
            valid_m = pl.min(TILE_M, M - m_base)
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
                pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
                pl.load(cur_a_l1, a, [m_base, 0])
                pl.load(cur_b_l1, b, [0, n_off])
                pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
                pl.move(cur_a_l0a, cur_a_l1)
                pl.move(cur_b_l0b, cur_b_l1)
                pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
                pl.store(workspace, cur_qk_l0c, [m_base, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector: compute Softmax on S in blocks along the N direction ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_off = mi * TILE_M + sub_id * VEC_ROWS
            if m_off < M:
                valid_rows = pl.min(VEC_ROWS, M - m_off)
                cur_qk = qk_vec.current()
                cur_tmp = tmp_vec.current()
                cur_exp = exp_vec.current()
                cur_red = red_vec.current()
                cur_red_rm = red_rm.current()
                cur_global_max = global_max.current()
                cur_global_max_rm = global_max_rm.current()
                cur_global_sum = global_sum.current()
                cur_global_sum_rm = global_sum_rm.current()
                pl.set_validshape(cur_red, [valid_rows, 1])
                pl.set_validshape(cur_red_rm, [1, valid_rows])
                pl.set_validshape(cur_global_max, [valid_rows, 1])
                pl.set_validshape(cur_global_max_rm, [1, valid_rows])
                pl.set_validshape(cur_global_sum, [valid_rows, 1])
                pl.set_validshape(cur_global_sum_rm, [1, valid_rows])

                # First pass: merge the row maxima of all N blocks.
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk, [valid_rows, valid_n])
                    pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                    pl.load(cur_qk, workspace, [m_off, n_off])
                    pl.maximum(cur_red, cur_qk, cur_tmp, dim=0)
                    if nj == 0:
                        pl.mul(cur_global_max_rm, cur_red_rm, 1.0)
                    else:
                        pl.maximum(cur_global_max_rm, cur_global_max_rm, cur_red_rm)

                # Second pass: accumulate the exp sums of all N blocks based on the global maximum value.
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk, [valid_rows, valid_n])
                    pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                    pl.set_validshape(cur_exp, [valid_rows, valid_n])
                    pl.load(cur_qk, workspace, [m_off, n_off])
                    pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                    pl.exp(cur_exp, cur_exp)
                    pl.sum(cur_red, cur_exp, cur_tmp, dim=0)
                    if nj == 0:
                        pl.mul(cur_global_sum_rm, cur_red_rm, 1.0)
                    else:
                        pl.add(cur_global_sum_rm, cur_global_sum_rm, cur_red_rm)

                # Third pass: normalize block by block and write the result directly back to out.
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk, [valid_rows, valid_n])
                    pl.set_validshape(cur_exp, [valid_rows, valid_n])
                    pl.load(cur_qk, workspace, [m_off, n_off])
                    pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                    pl.exp(cur_exp, cur_exp)
                    pl.expand_div(cur_exp, cur_exp, cur_global_sum, dim=0)
                    pl.store(out, cur_exp, [m_off, n_off])


# Host-side call
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(0)
M, N = 144, 1024
a = torch.randn(M, K_SIZE, device=device, dtype=torch.float16) * 0.1
b = torch.randn(K_SIZE, N, device=device, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, device=device, dtype=torch.float32)
workspace = torch.zeros(M, N, device=device, dtype=torch.float32)

matmul_softmax_kernel(a, b, out, workspace)
torch.npu.synchronize()

matmul_golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
golden = torch.softmax(matmul_golden, dim=-1)
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul-Softmax kernel passed!")
```

#### Implementation Notes

- The Tile settings `compact=1` in L0A Buffer, L0B Buffer, and L0C Buffer enable the M/N tail blocks to use compact strides based on the current valid sizes on the L1 Buffer→L0 Buffer and L0C Buffer→GM paths. The actual M/N ranges are passed on both sides through `pypto_pro.language.set_validshape`.
- Vector reduction produces a DN result of `[M half block, 1]`, which is used by `pypto_pro.language.expand_sub` and `pypto_pro.language.expand_div` for row-wise broadcasting. A `[1, M half block]` row-major view is simultaneously established at the same UB address for merging the maximum values and sum of exponentials across N blocks.
- `pypto_pro.language.jit(auto_mutex=True)` manages the intra-core mutex of each Tile, but the data readiness relationship between Cube and Vector is still established by `pypto_pro.language.system.set_cross_core` and `pypto_pro.language.system.wait_cross_core`.
- The host test scales the random inputs down by a factor of 0.1 to prevent the high saturation of Softmax from amplifying floating-point rounding differences. This scaling is used only for testing and is not part of the Kernel computation.

For the SIMD programming model, see [SIMD Programming Paradigm](../programming_paradigm/SIMD/programming_paradigm.md).

### VF API Implementation

VF (Vector Function) is defined using [`@pypto_pro.language.vector_function`](vector_computation/reg_computation.md##vf-functions-and-execution-domain) and reads/writes data between UB and vector registers through the [`vf.*` API](../../../../api/pro_api/SIMD-API/Reg-API/index.md). The following version reuses the preceding Cube computation, workspace layout, and cross-core synchronization, while the Vector side uses VF to compute Softmax. `[M half block, N block]` is first transposed in UB to `[N block, 64]`, so that each N position corresponds to one FP32 vector register, and the lanes in the register correspond to different M rows.

For basic concepts and API descriptions related to VF, see [Reg computation](vector_computation/reg_computation.md).

The data organization on the Vector side is as follows:

- `qk_nd` loads `[M half block, N block]` from the workspace and then transposes it to `qk_dn[N block, 64]`, so that each N position corresponds to a 64-lane FP32 register.
- `global_max` and `global_sum` store the per-row state across N blocks in UB, and the three passes update the maximum value, the sum of exponentials, and the normalized result, respectively.
- The M tail block restricts valid lanes through `vf.update_mask(valid_rows)`, and the N tail block restricts the actually processed positions through the VF loop upper bound `valid_n`.
- Before the state written by VF is read by subsequent VF or Vector operations, use [`vf.mem_bar`](../../../../api/pro_api/SIMD-API/Reg-API/data_movement/mem_bar.md) and set `mode=pypto_pro.language.MemBarMode.VST_VLD` to ensure local storage ordering; the Tile dependency between MTE and Vector is still managed by mutex.

```python
import os
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
import torch
import torch_npu

TILE_M = 64
TILE_N = 64
VEC_ROWS = 32
K_SIZE = 128
VF_LANES = 64
NEG_INF = -1e30

@pl.vector_function
def softmax_vf_init(global_max, global_sum, valid_rows: pl.DT_INT64):
    """Initialize the per-row maximum value and sum of exponentials saved across N blocks."""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    max_reg = vf.full(NEG_INF, preg, dtype=pl.DT_FP32)
    sum_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(global_max, max_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_update_max(src_dn, global_max,
                          valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """Merge the current N block into the per-row global maximum value."""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        max_reg = vf.max(max_reg, src_reg, preg)
    vf.store_align(global_max, max_reg, preg)


@pl.vector_function
def softmax_vf_update_sum(src_dn, global_max, global_sum,
                          valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """Accumulate the sum of exponentials of the current N block based on the global maximum value."""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        sum_reg = vf.add(sum_reg, exp_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_normalize(src_dn, dst_dn, global_max, global_sum,
                         valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """Normalize the current N block using the maximum and sum of exponentials of the full N axis."""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        out_reg = vf.div(exp_reg, sum_reg, preg)
        vf.store_align(dst_dn + ni * VF_LANES, out_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.jit(auto_mutex=True)
def matmul_softmax_vf_kernel(a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
                             b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
                             out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
                             workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    M = a.shape[0]
    N = b.shape[1]
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    M_TILES = (M + TILE_M - 1) // TILE_M
    N_TILES = (N + TILE_N - 1) // TILE_N

    # ---- Cube Tile: compute S = A @ B ----
    tt_a_mat = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_b_mat = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat)
    tt_left = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left,
                          compact=1)
    tt_right = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Right,
                           compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         compact=1)

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile: use 64x64 physical tiles before and after TTRANS, and use a mask in VF to narrow the M tail block ----
    tt_vf_block = pl.TileType(shape=[VF_LANES, TILE_N], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec)
    tt_vf_state = pl.TileType(shape=[1, VF_LANES], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec)

    qk_nd = pl.make_tile_group(type=tt_vf_block, addrs=0x0000, mutex_ids=[5])
    qk_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x4000, mutex_ids=[6])
    out_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x8000, mutex_ids=[7])
    out_nd = pl.make_tile_group(type=tt_vf_block, addrs=0xC000, mutex_ids=[8])
    global_max = pl.make_tile_group(type=tt_vf_state, addrs=0x10000, mutex_ids=[9])
    global_sum = pl.make_tile_group(type=tt_vf_state, addrs=0x10100, mutex_ids=[10])

    # ==== Cube: split cores along the M axis and split blocks along the N axis, then write the result to workspace ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_base = mi * TILE_M
            valid_m = pl.min(TILE_M, M - m_base)
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
                pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
                pl.load(cur_a_l1, a, [m_base, 0])
                pl.load(cur_b_l1, b, [0, n_off])
                pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
                pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
                pl.move(cur_a_l0a, cur_a_l1)
                pl.move(cur_b_l0b, cur_b_l1)
                pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
                pl.store(workspace, cur_qk_l0c, [m_base, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector: invoke VF on the transposed [N block, 64] UB tile ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        for mi in pl.range(core_id, M_TILES, num_cores):
            m_off = mi * TILE_M + sub_id * VEC_ROWS
            if m_off < M:
                valid_rows = pl.min(VEC_ROWS, M - m_off)
                cur_qk_nd = qk_nd.current()
                cur_qk_dn = qk_dn.current()
                cur_out_dn = out_dn.current()
                cur_out_nd = out_nd.current()
                cur_global_max = global_max.current()
                cur_global_sum = global_sum.current()
                pl.set_validshape(cur_global_max, [1, valid_rows])
                pl.set_validshape(cur_global_sum, [1, valid_rows])
                softmax_vf_init(cur_global_max, cur_global_sum, valid_rows)

                # First pass: convert each N block to [N block, 64] and update the per-row maximum lane by lane.
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                    pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                    pl.load(cur_qk_nd, workspace, [m_off, n_off])
                    pl.transpose(cur_qk_dn, cur_qk_nd)
                    softmax_vf_update_max(cur_qk_dn, cur_global_max, valid_rows, valid_n)

                # Second pass: accumulate the sum of exponentials of the full N axis lane by lane based on the global maximum value.
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                    pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                    pl.load(cur_qk_nd, workspace, [m_off, n_off])
                    pl.transpose(cur_qk_dn, cur_qk_nd)
                    softmax_vf_update_sum(cur_qk_dn, cur_global_max, cur_global_sum,
                                          valid_rows, valid_n)

                # Third pass: after VF normalization, convert back to [M half block, N block] and write back to GM.
                for nj in pl.range(0, N_TILES, 1):
                    n_off = nj * TILE_N
                    valid_n = pl.min(TILE_N, N - n_off)
                    pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                    pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                    pl.set_validshape(cur_out_dn, [valid_n, valid_rows])
                    pl.set_validshape(cur_out_nd, [valid_rows, valid_n])
                    pl.load(cur_qk_nd, workspace, [m_off, n_off])
                    pl.transpose(cur_qk_dn, cur_qk_nd)
                    softmax_vf_normalize(cur_qk_dn, cur_out_dn, cur_global_max,
                                         cur_global_sum, valid_rows, valid_n)
                    pl.transpose(cur_out_nd, cur_out_dn)
                    pl.store(out, cur_out_nd, [m_off, n_off])


# Host-side invocation
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(0)
M, N = 144, 1024
a = torch.randn(M, K_SIZE, device=device, dtype=torch.float16) * 0.1
b = torch.randn(K_SIZE, N, device=device, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, device=device, dtype=torch.float32)
workspace = torch.zeros(M, N, device=device, dtype=torch.float32)

matmul_softmax_vf_kernel(a, b, out, workspace)
torch.npu.synchronize()

matmul_golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
golden = torch.softmax(matmul_golden, dim=-1)
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul-Softmax VF kernel passed!")
```

#### Implementation Notes

- An FP32 VF register contains 64 lanes. Each AIV processes up to 32 M rows, and `vf.update_mask(valid_rows)` enables only the corresponding lanes. The 64-column physical width aligns the start address of each N position to a full register.
- The UB Tiles before and after the transpose are both declared as 64×64. The actual transpose range of `TTRANS` is derived from the `valid_shape` of the source Tile: the valid regions of the source and destination are `[valid_rows, valid_n]` and `[valid_n, valid_rows]`, respectively. The VF loop upper bound and predicate then handle the N/M tail blocks separately.
- `vf.mem_bar(mode=pypto_pro.language.MemBarMode.VST_VLD)` handles the local memory dependency between the VF/Vector store and the subsequent Vector load, including reading `global_max`/`global_sum` across N blocks and reading the normalized results in `transpose`. The Tile pipeline dependencies among MTE2, MTE3, and VF are still managed by `auto_mutex=True`.

## Advanced Example: High-Performance VF Implementation of FlashAttention

In the preceding basic example, the Matmul result is first written to the workspace in GM and then read by the Vector side. In the high-performance FlashAttention implementation, the computation $O=\operatorname{Softmax}(QK^T/\sqrt{D})V$ is performed with intermediate results passed between AIC and AIV through the on-chip CV path, and only the final output is written back to GM.

The main logic of this implementation is as follows:

1. The Cube side computes $QK^T$ in blocks and moves the result from the L0C Buffer to UB, handing it over to the Vector side for processing.
2. The Vector side uses VF to perform block-wise Softmax and online statistics updates, handling the tail blocks in the M and N directions through masking.
3. The Vector side writes the Softmax probabilities from UB to the L1 Buffer, and the Cube side then computes $PV$; the Vector side receives the result and completes the online output update and final write-back.
4. Multiple Tile groups rotate through the Tile Group, with `auto_mutex=True` managing intra-core Tile access; between AIC and AIV, `set_cross_core` and `wait_cross_core` are used to establish dependencies for data readiness and Buffer slot release, and prefetching is used to overlap the QK computation with the subsequent PV and Vector stages.

For the complete Kernel, VF helper functions, on-chip address and synchronization ID configuration, as well as the execution and accuracy verification code, see [FlashAttention High-Performance VF Implementation ST Test Case](../../../../../../python/tests/st/pypto_pro/frontend/fa/test_fa_perf_tkv_preload_dn_vf_tail_tilegroup.py).
