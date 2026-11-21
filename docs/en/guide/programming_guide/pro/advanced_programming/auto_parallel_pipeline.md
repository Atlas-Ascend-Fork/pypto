# Automatic CV Parallel Pipeline

<!-- md-trans-meta sourceCommit=6d462eb91e1be85bc92f842fd449c778962e7249 translatedAt=2026-11-15T02:06:49.158Z pushedAt=2026-11-18T06:03:01.338Z nl=2 -->

## Description

In a CV fusion operator, the Cube and Vector computations depend on each other. If they are executed as a serial pipeline, one core can only idle while the other core is working. To improve the parallelism between Cube and Vector, the upstream core computes several iterations ahead, while the downstream core processes the data of earlier iterations that the upstream core has already completed. The two cores work simultaneously at staggered rhythms, thereby hiding each other's waiting time.

Automatic CV parallel pipeline is a compile-time transformation in pypto_pro.language.jit. It provides two capabilities:

- **Automatic pipeline scheduling**: automatically rewrites the hand-written serial pipeline kernel code of a CV fusion operator into a parallel pipeline version to improve performance;
- **Automatic inter-core synchronization**: the serial version developed by the user does not need any hand-written CV inter-core synchronization instructions, and the parallel pipeline version inserts all required inter-core synchronization.

The following figure compares the execution diagrams of the same operator (4 stages, with Cube/Vector alternating) under the serial pipeline and the parallel pipeline (Cube executed 2 iterations ahead):

**Figure 1** Execution rhythm comparison between the serial pipeline and the parallel pipeline

![Execution rhythm comparison between the serial pipeline and the parallel pipeline](../../../figures/pro/pro_parallel_pipeline_serial_vs_parallel.svg "Execution rhythm comparison between the serial pipeline and the parallel pipeline")

In the figure, `sN·iM` indicates that stage N is processing the data of iteration M, and the width of each cell represents the duration of that stage (the duration of each stage differs, and the values in the figure are illustrative).

Note that the upper limit of the pipelining benefit is determined by **the slower core**: in the figure, the per-iteration duration on the Vector side is higher than that on the Cube side, so the steady-state rhythm is determined by the Vector side, and the Cube side has waiting gaps. The more balanced the stage division (the closer the durations of the two cores), the fuller the pipeline is filled and the higher the benefit. In addition, establishing and draining the pipeline each require several beats. The more iterations there are, the lower the proportion of this overhead.

## Usage

The usage constraints of the automatic CV parallel pipeline are listed in the "Notes" following each usage step. Check the corresponding step during development.

### Writing the Stage Function

You need to divide the operator into several computation flows, each corresponding to a stage function identified by the @pypto_pro.language.pipeline.stage decorator.

```python
import pypto_pro.language as pl

@pl.pipeline.stage
def stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db):
    """Cube: mm1 = A_i @ B"""
    cur_a = a_l1_db.next()
    pl.load(cur_a, a, [ki * TILE, 0])
    ...  # Ordinary Tile/Buffer operations; no synchronization needs to be written
```

> [!NOTE]
>
> - A stage function cannot nest another stage and cannot have a return value. An ordinary function called by a stage cannot return a Tile or Tile Group.
> - The parameters of a stage function support scalars (integers, floating-point numbers, and booleans), fixed-length one-dimensional scalar arrays, Tensor, Tile, Tile Group, ordinary tuples, the return value of `pypto_pro.language.make_tuple()`, and structs declared by `pypto_pro.language.struct()`. The fields of a struct support only scalars and fixed-length one-dimensional scalar arrays. `pypto_pro.language.struct_array()` is not supported. The values in `pypto_pro.language.make_tuple()` and ordinary tuples must be determined outside the pipeline loop and must not change across iterations within the loop.

### Declaring Cross-Core Shared Buffers

Cross-core shared Buffers fall into two categories: one uses the UB or L1 Buffer private to each core and is declared through the make_tile_group API; the other uses the GM (workspace) shared by all cores and is declared through the [pypto_pro.language.make_tensor](../../../../api/pro_api/SIMD-API/resource_management/make_tensor.md) API. The inter-core synchronization rules for the two categories are exactly the same; the only difference lies in how the Buffer itself is allocated, as described in the following two sections respectively. This section collectively refers to each position used in rotation as a slot: in UB and L1 Buffers, a slot is a Tile; in GM, a slot is a contiguous space divided equally by `depth`.

`mutex_ids` and `fwd_ids`/`bwd_ids` act on different scopes: `mutex_ids` is used to manage the pipeline synchronization and address reuse of Tiles within a core, and only UB and L1 Buffers need to configure it; `fwd_ids` and `bwd_ids` are used for data handover between Cube and Vector. Only when `fwd_ids` or `bwd_ids` is declared and the Buffer is used by a Cube stage and a Vector stage respectively does the framework recognize it as a cross-core shared Buffer and insert the corresponding inter-core synchronization.

The number of `fwd_ids` and `bwd_ids` is not required to always match the number of slots: you can configure one id for each slot, or configure only one id shared by all slots. Sharing a single inter-core synchronization id serializes the handover between different slots, which reduces the number of synchronization ids used but may lower the pipeline parallelism.

Here, `fwd_ids` is forward synchronization (the producer notifies the consumer after writing), where a set is inserted after the producer stage and a wait is inserted before the consumer stage; `bwd_ids` is backward synchronization (the consumer notifies the producer that it can overwrite after finishing use), where a set is inserted after the consumer stage and a wait is inserted before the producer stage. When only `fwd_ids` is configured, it only guarantees that the consumer reads the data written by the producer, but does not guarantee that the consumer has finished reading before the producer overwrites in the next round. The slot used in each iteration rotates by the number of slots; the inter-core synchronization id rotates by `ids[iteration index % len(ids)]`, so the rotation periods of the two can differ.

> [!NOTE]
>
> - Cross-core Buffers and intra-core Buffers must be declared in the kernel function body, outside all loops, and can also be returned by an ordinary function called in the kernel function body. They cannot be written inside a stage, nor inside an ordinary function called by a stage.
> - The return value of a declaration must be bound to a variable in the kernel function body and cannot be passed directly as an argument to a call.
> - Each Buffer must use a different variable name, and names cannot be duplicated across different sections.
> - Automatic pipelining does not scan Tiles declared by [pypto_pro.language.make_tile](../../../../api/pro_api/SIMD-API/resource_management/make_tile.md), and does not insert any synchronization for the ops that access them. The framework also does not check their addresses. When their address range overlaps with a cross-core Buffer (or a Buffer that reuses the address of a cross-core Buffer), no error is reported, but no synchronization is inserted either. Developers must ensure that the addresses do not conflict. For Buffers participating in the pipeline, it is recommended to declare them uniformly with make_tile_group. When rotation and automatic synchronization are not needed, configure only `depth` and not `mutex_ids`.
> - A cross-core Buffer must be used by exactly two adjacent stages, which are located on Cube and Vector respectively and form a one-to-one producer-consumer relationship. When data needs to be passed to a later stage, it should be handed over stage by stage.
> - The value range of `fwd_ids` and `bwd_ids` is 0 to 15. They can only be written directly as a compile-time constant integer list, or use a variable bound to an integer list outside the kernel. Expressions such as slicing, concatenation, and function calls are not supported. The number of IDs must equal the number of slots or equal 1. When it equals 1, all slots share the same synchronization ID, which may reduce performance.
> - The slots of a cross-core Buffer must rotate in iteration order. Stages are numbered in the order they are written in the main loop (regardless of whether the first stage is on Cube or Vector). When the stage that writes the Buffer has an odd number, at least 2 slots are required; when it has an even number, at least `preload` slots are required. If the number of slots is insufficient, a compile-time error is reported with a hint of the minimum number of slots required.
> - When space is sufficient, you can allocate more slots than the lower limit and determine whether there is a performance benefit based on actual measurement results. When adjusting the number of slots, you need to adjust the number of `fwd_ids` and `bwd_ids` and the space occupied by the Buffer accordingly, except when the inter-core synchronization ID is configured as a single ID shared by all slots.

#### Using UB or L1 Buffer

`mutex_ids` must be configured with an ID for each Tile in the Buffer, so its quantity is consistent with the number of Tiles. `target_memory` only declares the physical storage space where the Buffer resides and does not indicate whether it is cross-core; the cross-core shared Buffer declared by make_tile_group supports UB and L1 Buffer. The following `mm1_vec_db` is an example in which the Cube stage passes the computation result to the Vector stage; in the complete example later, `p_mat_db` is written by the Vector stage and then passed to the Cube stage for reading. The cross-core relationship of both is determined by the producer/consumer relationship of the stages and by `fwd_ids` and `bwd_ids`.

```python
import pypto_pro.language as pl

mm1_vec_db = pl.make_tile_group(
    type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
    addrs=0x0000,
    mutex_ids=[12, 13],
    fwd_ids=[0, 1],
    bwd_ids=[2, 3],
)
```

> [!NOTE]
>
> - When reusing addresses, at most two Buffers are allowed to be reused, and the number of Tiles and `mutex_ids` of both must be consistent; different `mutex_ids` cannot form address mutual exclusion.
> - When adjusting the number of Tiles, you also need to synchronously adjust the number of `mutex_ids` and the address space occupied by the Buffer.

#### Using GM

When the core-private UB and L1 buffer space is insufficient, GM can be used to hand over data. Add `depth`, `fwd_ids`, and `bwd_ids` to `make_tensor`, and the framework identifies the Tensor view as a cross-core shared buffer.

`shape` and `stride` describe a **single slot**, and `depth` is the number of slots. The element offset of the k-th slot relative to the declaration position is k × the number of elements in a single slot, where the number of elements in a single slot is `max(shape[i] × stride[i])`, or the product of all dimensions of `shape` when `stride` is omitted. At each stage call point, the framework selects the corresponding slot based on the task number of the current beat and passes it to the stage. Therefore, the stage function body is written for a single slot, consistent with the way a single Tile is taken out of a UB or L1 buffer.

GM is shared by all cores, whereas UB and L1 buffers are private to each core. Therefore, GM needs to be divided into independent regions for each core, and the offset of that region is written to the first address of the declaration. The slot offset calculated by the framework is added on top of this first address. Cube and Vector must use the same core number, so the core number needs to be divided by the number of sub-cores of Vector.

```python
import pypto_pro.language as pl

# Each core occupies WS_PER_CORE bytes, where the first GM_DEPTH slots are for mm1_gm
core_id = pl.get_block_idx() // pl.get_subblock_num()
ws_core = pl.addptr(ws, core_id * WS_PER_CORE)
mm1_gm = pl.make_tensor(
    ws_core, [TILE, TILE], dtype=pl.DT_FP32,
    depth=2,
    fwd_ids=[0, 1],
    bwd_ids=[2, 3],
)
```

> [!NOTE]
>
> - `depth` must be a positive compile-time integer. When `fwd_ids` or `bwd_ids` is configured, `depth` must be configured. When automatic pipelining is not enabled, these three parameters are ignored.
> - `shape` and `stride` need to be written as lists: you can write a list directly, or use a variable bound to a list inside or outside the kernel function body. The value of each dimension can be a runtime integer expression, in which case the slot offset is calculated at runtime. If it cannot be resolved into a list, a compile-time error is reported because the number of dimensions is unknown and the size of a single slot cannot be determined.
> - A GM cross-core buffer has no `mutex_ids` and does not need to be configured.
> - Unlike a Tile, GM does not support sliced subviews. In the stage function body, access the data in a slot through the element coordinates of load and store. For details, see [Tensor Creation and Operations](../development/tensor_creation_and_operations.md#data-movement-between-tensor-and-tile).
> - A GM cross-core buffer does not support address reuse.

### Writing the Main Loop

In the main loop, stage functions are written in dependency order, and the stages must alternate between CV.

```python
import pypto_pro.language as pl

for ki in pl.range(0, N_ITER):
    with pl.section_cube():
        stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db)
    with pl.section_vector():
        stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db)
    with pl.section_cube():
        stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db)
    with pl.section_vector():
        stage4(ki, sub_id, out, out_vec_db)
```

Multiple independent for loops can be written in a single kernel. Each loop undergoes its own pipeline transformation without affecting the others. The multiple loops execute sequentially in source code order, and preload can be configured per loop (see the next section). The full-core synchronization between two adjacent pipelines is automatically inserted by the framework and does not need to be written manually.

```python
import pypto_pro.language as pl

for ki in pl.range(0, N_ITER):        # First pipeline
    with pl.section_cube():
        stage1(ki, ...)
    with pl.section_vector():
        stage2(ki, ...)

# The full-core synchronization here is automatically inserted by the framework and does not need to be written manually

for kj in pl.range(0, M_ITER):        # Second pipeline
    with pl.section_cube():
        stage3(kj, ...)
    with pl.section_vector():
        stage4(kj, ...)
```

> [!NOTE]
>
> - Stage calls of the same pipeline must be placed in the same for loop. The loop step must be positive, and reverse iteration is not supported. Each `section_cube()` or `section_vector()` block can contain only one stage call, and stages must be strictly alternated between Cube and Vector. No other statements can be inserted between the first stage and the last stage.
> - Cross-core Buffers, and intra-core Buffers that reuse addresses with cross-core Buffers, cannot be directly obtained or operated on inside the pipeline loop body. These operations should be placed in stage functions. The same stage function can be called only once in a pipeline, and different stage functions cannot have the same name.
> - An if statement can be used to control whether a stage is executed, but the condition must be a compile-time constant. When calling `group.next()` outside the pipeline loop to obtain a Tile, a separate assignment statement is required and it cannot be embedded in other expressions. The same variable can obtain a Tile only once; use multiple variables when multiple Tiles are needed.
> - A kernel can contain multiple independent pipeline loops, but these loops cannot be nested in the same outer loop. Full-core synchronization between loops is automatically inserted by the framework. Do not use framework-reserved variable names starting with `_pl_` inside the kernel.

### Enabling Pipeline Transformation

In the pypto_pro.language.jit API, configure it through the PipelineConfig parameter:


| Parameter    | Description                                                                                                                                                                                                      |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| preload | Indicates the number of iterations that the upstream core computes in advance. The type is int or list[int]. When a single integer is passed, all pipeline loops in the kernel share this value. When a list is passed, each pipeline loop is configured separately in source code order. A loop with a value of 0 is not rewritten into a pipeline; only inter-core synchronization is inserted into the original serial loop. |

**Recommended process**: First configure **preload=0** and run to confirm that the serial version is accurate, then gradually increase preload to enable the pipeline. While increasing preload, check whether the number of Tiles of each cross-core Buffer is sufficient according to the instructions following [Declaring Cross-Core Shared Buffers](#declaring-cross-core-shared-buffers). When there are multiple pipeline loops, you can set only one of them to 0 (for example, preload=[0, 2]) to verify the serial accuracy of that pipeline separately.

During tuning, observe the impact of both `preload` and the number of Buffer Tiles on performance. If performance does not improve after increasing `preload` or adding Tiles, reduce the number of Tiles to the minimum value corresponding to the current `preload` to save on-chip storage space. After the reduction, the lower limit of the number of slots specified in the instructions following [Declaring Cross-Core Shared Buffers](#declaring-cross-core-shared-buffers) must still be met, and accuracy and performance must be re-verified.

> [!NOTE]
>
> - `preload` must be greater than or equal to 0. When a list is passed, the list length must match the number of pipeline loops in the kernel.

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=0))
def pipeline_demo_kernel(...):
    ...
```

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))
def pipeline_demo_kernel(...):
    ...
```

```python
import pypto_pro.language as pl

# The kernel contains two pipeline loops: the first has preload=1, and the second has preload=2
@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=[1, 2]))
def pipeline_demo_kernel(...):
    ...
```

### Viewing the Generated Code

The parallel pipeline code automatically generated by the framework is saved in the compilation output directory under the file name pipeline_generated.py. You can continue to modify and debug based on this code.

## Examples

```python
import os

import pypto_pro.language as pl
import pytest
import torch
import torch_npu

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE = 64
TILE_HALF = TILE // 2  # dual mode splits along M, and each AIV processes half
N_ITER = 4


@pl.pipeline.stage
def stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db):
    """Cube: mm1 = A_i @ B"""
    cur_a = a_l1_db.next()
    pl.load(cur_a, a, [ki * TILE, 0])
    b_slot = b_l1.current()
    left = left_db.next()
    right = right_db.next()
    acc = acc_db.next()
    pl.move(left, cur_a)
    pl.move(right, b_slot)
    pl.matmul(acc, left, right)
    mm1_vec = mm1_vec_db.next()
    # DualModeSplitM: the acc of [TILE, TILE] is split along M, and each AIV takes [TILE_HALF, TILE]
    pl.move(mm1_vec, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db):
    """Vector: apply relu to this AIV's half, convert to NZ, and insert it back into the L1 Buffer by row offset"""
    mm1_vec = mm1_vec_db.next()
    relu_vec = relu_vec_db.next()
    pl.relu(relu_vec, mm1_vec)
    relu_nz = relu_nz_db.next()
    pl.move(relu_nz, relu_vec)  # ND -> NZ, insert requires the source Tile to be NZ
    p_mat = p_mat_db.next()
    pl.insert(p_mat, relu_nz, [sub_id * TILE_HALF, 0])


@pl.pipeline.stage
def stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db):
    """Cube: mm2 = relu(mm1) @ D"""
    d_slot = d_l1.current()
    p_mat = p_mat_db.next()
    left = left_db.next()
    right = right_db.next()
    acc = acc_db.next()
    pl.move(left, p_mat)
    pl.move(right, d_slot)
    pl.matmul(acc, left, right)
    out_vec = out_vec_db.next()
    pl.move(out_vec, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def stage4(ki, sub_id, out, out_vec_db):
    """Vector: Each AIV writes back half of the result of this iteration."""
    out_vec = out_vec_db.next()
    pl.store(out, out_vec, [ki * TILE + sub_id * TILE_HALF, 0])


@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))
def pipeline_demo_kernel(
    a: pl.Tensor[[N_ITER * TILE, TILE], pl.DT_FP32],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    d: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    out: pl.Tensor[[N_ITER * TILE, TILE], pl.DT_FP32],
):
    # ===== Cross-core shared Buffer: Declare fwd_ids/bwd_ids, and the framework automatically inserts synchronization based on them =====
    mm1_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[12, 13], fwd_ids=[0, 1], bwd_ids=[2, 3],
    )
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x8000, mutex_ids=[14, 15], fwd_ids=[4, 5], bwd_ids=[6, 7],
    )
    out_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000, mutex_ids=[16, 17], fwd_ids=[8, 9], bwd_ids=[10, 11],
    )

    # ===== Cube-side local Buffer =====
    with pl.section_cube():
        a_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x10000, mutex_ids=[2],
        )
        d_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x14000, mutex_ids=[3],
        )
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[4, 5],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0000, mutex_ids=[6, 7],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024,
            ),
            addrs=0x0000, mutex_ids=[8, 9, 10, 11],
        )
        # Move B and D in once outside the loop and reuse them throughout
        b_slot = b_l1.current()
        d_slot = d_l1.current()
        pl.load(b_slot, b, [0, 0])
        pl.load(d_slot, d, [0, 0])

    # ===== Vector-side local Buffer =====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        relu_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x8000, mutex_ids=[18, 19],
        )
        relu_nz_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE_HALF, TILE], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
            ),
            addrs=0x18000, mutex_ids=[20, 21],
        )

    # ===== Pipeline loop: Call the 4 stages alternately by Cube/Vector =====
    for ki in pl.range(0, N_ITER):
        with pl.section_cube():
            stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db)
        with pl.section_vector():
            stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db)
        with pl.section_cube():
            stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db)
        with pl.section_vector():
            stage4(ki, sub_id, out, out_vec_db)


@pytest.mark.soc("950")
def test_pipeline_demo_kernel():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    a = torch.randn(N_ITER * TILE, TILE, device=device, dtype=torch.float32)
    b = torch.randn(TILE, TILE, device=device, dtype=torch.float32)
    d = torch.randn(TILE, TILE, device=device, dtype=torch.float32)
    out = torch.zeros(N_ITER * TILE, TILE, device=device, dtype=torch.float32)

    pipeline_demo_kernel[None, 1](a, b, d, out)
    torch.npu.synchronize()

    ref = torch.relu(a @ b) @ d
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
```

### Pipeline and Synchronization Relationship of the Complete Example

The following figure corresponds to the `N_ITER=4` and `preload=2` configuration of the complete example above. `sN·iM` indicates that stage N is processing the M-th iteration. The two AIVs execute the same Vector stage and each processes half of the M direction, so they are merged into a single Vector lane in the figure. The box widths and gaps are used to illustrate the execution time and waiting relationship of each stage, and do not represent actual execution cycles.

Cross-core synchronization IDs rotate according to the Buffer slots corresponding to logical iterations, rather than being assigned by AIV number. Taking `mm1_vec_db` as an example, `stage1(i0)` notifies `stage2(i0)` through Forward ID 0, and `stage1(i1)` notifies `stage2(i1)` through Forward ID 1. Subsequently, `i2` and `i3` reuse slot 0 and slot 1, respectively. When the two AIVs execute the same Vector stage iteration, they use the same cross-core event corresponding to that iteration. The default `INTRA_BLOCK` synchronization mode handles the handover between the AIC and the two AIVs.

**Figure 2** Execution order and synchronization relationship of the complete example when preload=2

![Pipeline and synchronization relationship of the complete example of automatic CV parallel pipeline](../../../figures/pro/pro_auto_parallel_pipeline_example.svg "Pipeline and synchronization relationship of the complete example of automatic CV parallel pipeline")

- **Parallel preload**: During the fill phase, the preceding stages first process two iterations. After entering the steady state, the Cube side interleaves `stage1(i+2)` and `stage3(i)`, while the Vector side interleaves `stage2(i+2)` and `stage4(i)`. The AIC and AIVs process data of different iterations simultaneously. After the main loop ends, the compiler continues to drain the `stage3` and `stage4` tasks that have already entered the pipeline.
- **Inter-core synchronization**: The compiler automatically inserts `set_cross_core` and `wait_cross_core` based on the `fwd_ids` and `bwd_ids` of the cross-core Buffers. `mm1_vec_db` uses Forward IDs 0/1 and Backward IDs 2/3, `p_mat_db` uses Forward IDs 4/5 and Backward IDs 6/7, and `out_vec_db` uses Forward IDs 8/9 and Backward IDs 10/11. Forward events ensure that the consumer waits for the data to be written, while Backward events ensure that the producer waits for the consumer to finish using the data before overwriting the corresponding slot.
- **Intra-core synchronization**: `auto_mutex=True` manages, based on the `mutex_ids` of each Tile Group, the read/write of Tiles, slot rotation, and address reuse among different pipeline units within the same AIC or AIV. `mutex_ids` handles intra-core synchronization, while `fwd_ids` and `bwd_ids` handle data handover between the AIC and AIVs. Their scopes of effect are different.

In this example, `mm1_vec_db` and `out_vec_db` are written by the odd-numbered stage1 and stage3, respectively, so at least 2 Tiles are required. `p_mat_db` is written by the even-numbered stage2, so its Tile count cannot be less than `preload`. Since `preload=2` here, all three cross-core Buffers are configured with 2 Tiles.

> [!NOTE]
>
> - This example is designed for a single 1:2 CV execution group, so `block_dim` is set to 1. When extending to multiple execution groups, use `pypto_pro.language.get_block_idx()` to divide the GM data and output range processed by each execution group. For a complete description of `block_dim`, see [Kernel Function](../development/kernel_function.md#meaning-and-setting-of-blockdim).
