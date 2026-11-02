# Advanced Cube Computation

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T01:45:18.901Z pushedAt=2026-10-27T04:13:41.451Z nl=2 -->

In regular Cube computation, the framework automatically handles pipeline synchronization. For scenarios that require fine-grained control over the Cube computation pipeline and the Fixpipe data movement pipeline, such as K-dimension block accumulation and staged data movement of L0C Buffer results, you can use the `phase` parameter to synchronize via the hardware `unit_flag`. This section describes the applicable scenarios, supported APIs, synchronization semantics, usage constraints, and code examples of this advanced mechanism.

## Phase Usage Scenarios

During Matmul computation, if the Cube computation pipeline writes results to the L0C Buffer and the data in the L0C Buffer subsequently needs to be moved to GM, UB, or L1 Buffer through Fixpipe, you can configure AccPhase and STPhase to use the hardware unit_flag to synchronize Cube and Fixpipe.

Compared with the software synchronization automatically inserted by the framework, configuring `phase` reduces the software synchronization overhead between the Matmul computation pipeline and Fixpipe on the L0C Buffer, improving pipeline parallelism.

Typical usage scenarios include:

- K-dimension tiled Matmul, where results of multiple computations are accumulated into the same L0C Buffer block before being moved out.
- Matmul results need to be moved to UB for further post-processing by the Vector core, for example, the Softmax computation in Flash Attention.
- Other scenarios that require using the hardware unit_flag to synchronize the Cube computation pipeline and the Fixpipe data movement pipeline.

For a single Matmul computation, if there is no need to use the hardware `unit_flag` for pipeline synchronization, you can leave `phase` unconfigured and let the framework automatically insert software synchronization.

## Phase Mechanism

The `phase` parameter of the Matmul series of APIs uses `pypto_pro.language.AccPhase`, and the `phase` parameter of the Fixpipe data movement APIs uses `pypto_pro.language.STPhase`.

Together, they control the hardware `unit_flag` handshake between the Cube computation pipeline and the Fixpipe data movement pipeline.

For details about the related enums, see:

- [`pypto_pro.language.AccPhase`](../../../../api/pro_api/SIMD-API/basic_data_structures/AccPhase.md)
- [`pypto_pro.language.STPhase`](../../../../api/pro_api/SIMD-API/basic_data_structures/STPhase.md)

The APIs that support configuring AccPhase include:

- [`pypto_pro.language.matmul`](../../../../api/pro_api/SIMD-API/cube_computation/matmul.md)
- [`pypto_pro.language.matmul_acc`](../../../../api/pro_api/SIMD-API/cube_computation/matmul_acc.md)
- [`pypto_pro.language.matmul_mx`](../../../../api/pro_api/SIMD-API/cube_computation/matmul_mx.md)
- [`pypto_pro.language.matmul_mx_acc`](../../../../api/pro_api/SIMD-API/cube_computation/matmul_mx_acc.md)

The APIs that support configuring STPhase include:

- [`pypto_pro.language.store`](../../../../api/pro_api/SIMD-API/memory_data_movement/store.md)
- [`pypto_pro.language.store_tile`](../../../../api/pro_api/SIMD-API/memory_data_movement/store_tile.md)
- [`pypto_pro.language.move`](../../../../api/pro_api/SIMD-API/memory_data_movement/move.md)
- [`pypto_pro.language.insert`](../../../../api/pro_api/SIMD-API/memory_data_movement/insert.md)

### AccPhase

When the Matmul series of APIs configures pypto_pro.language.AccPhase.Partial or pypto_pro.language.AccPhase.Final, the hardware unit_flag feature is enabled.

For Cube writing data to the L0C Buffer:

- unit_flag = 0: The hardware is allowed to write to the L0C Buffer.
- unit_flag = 1: The write operation is paused until unit_flag becomes 0 again.

The behavior of Partial and Final is as follows.

| Mode | Check unit_flag | Set unit_flag |
| --- | --- | --- |
| Partial | Yes, wait for unit_flag = 0 before writing | No, unit_flag is not changed |
| Final | Yes, wait for unit_flag = 0 before writing | Yes, set unit_flag to 1 after the write is complete |

Therefore, in a scenario with multiple Matmul accumulations, intermediate computations typically use Partial, and the final computation that writes to the L0C Buffer uses Final, notifying Fixpipe that it can begin reading data.

### STPhase

When the Fixpipe data-out API is configured with pypto_pro.language.STPhase.Partial or pypto_pro.language.STPhase.Final, the hardware unit_flag feature is also enabled.

For Fixpipe reading data from the L0C Buffer:

- unit_flag = 1: The hardware is allowed to read the L0C Buffer.
- unit_flag = 0: The read operation is paused until unit_flag becomes 1.

The behavior of Partial and Final is as follows.

| Mode | Check unit_flag | Set unit_flag |
| --- | --- | --- |
| Partial | Yes, wait for unit_flag = 1 before reading | No, does not change unit_flag |
| Final | Yes, wait for unit_flag = 1 before reading | Yes, sets unit_flag to 0 after the read completes |

After the STPhase.Final read completes, unit_flag is restored to 0, allowing the same L0C Buffer block to continue being used by subsequent Matmul computations.

### Relationship Between Phase and Automatic Synchronization

Whether `phase` is configured affects the synchronization method used by the framework between the Matmul computation pipeline and Fixpipe.

| Configuration | L0C Buffer Synchronization | Synchronization Mechanism |
| --- | --- | --- |
| With `phase` configured | No software synchronization is inserted between the Matmul computation pipeline and Fixpipe | Synchronization between the Matmul computation pipeline and Fixpipe is completed through the hardware unit_flag; the automatic synchronization of other tiles such as L1 Buffer, L0A Buffer, and L0B Buffer is not affected |
| Without `phase` configured | Software synchronization is inserted between the Matmul computation pipeline and Fixpipe | The framework ensures the execution order of the Matmul computation pipeline and Fixpipe through software synchronization |

Configuring `phase` only affects the synchronization of the L0C Buffer between the Matmul computation pipeline and Fixpipe, and does not affect the automatic synchronization mechanism of other buffers.

## Phase Usage Constraints

Improper `phase` configuration may lead to incorrect computation results or device hang. The following constraints must be met when using it:

1. **AccPhase and STPhase must be used together.**

   If the Matmul series of APIs is configured with phase, the corresponding store, store_tile, move, or insert that reads data from this L0C Buffer must also be configured with phase correctly.

2. **The last write and read must use the Final mode.**

   For the same L0C Buffer, the last write operation of the Matmul series of APIs must use AccPhase.Final, and the corresponding last read operation of Fixpipe must use STPhase.Final.

## Phase Usage Examples

### Single matmul (without K-dimension accumulation)

If phase is not passed, the framework automatically inserts synchronization:

```python
pl.matmul(ac, al, br)
pl.store(out, ac, [0, 0])
```

### K-dimension Block Accumulation (Multiple Blocks)

First block uses Partial, middle blocks use Partial, last block uses Final, and store uses STPhase.Final:

```python
with pl.section_cube():
    ac = acc.current()
    for k in pl.range(0, K_TOTAL, TILE_K):
        ...
        if k == 0:
            pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)        # First block
        elif k < K_TOTAL - TILE_K:
            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial) # Middle block
        else:
            pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)   # Last block
    pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)                # Final wrap-up
```

### Flash Attention Example

When the matmul result requires post-processing by the Vector core (such as softmax), you must use move to transfer the accumulator data to UB. store can only write directly to GM and cannot perform subsequent computation on UB. Take the QK matmul of Flash Attention as an example:

```python
with pl.section_cube():
    ac = acc.current()
    for k in pl.range(0, K_TOTAL, TILE_K):
        ...
        if k == 0:
            pl.matmul(ac, q, k, phase=pl.AccPhase.Partial)
        else:
            pl.matmul_acc(ac, ac, q, k, phase=pl.AccPhase.Final)
    # Move to UB for the Vector core to perform softmax (store cannot do L0C→UB)
    pl.move(qk_vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN,
            phase=pl.STPhase.Final)
with pl.section_vector():
    # softmax: row max → sub → exp → sum → scale
    pl.maximum(reduce_max, qk_vec, tmp, dim=1)
    pl.expand_sub(tmp, qk_vec, reduce_max, dim=1)
    # ... exp / sum / scale ...
    pl.store(p_buf, qk_vec, [...])
```

## Common Phase Errors

### Error Case 1: matmul Without Final Causes Hang

```python
pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)
```

**Symptom**: Hang.

**Cause**: matmul uses Partial, which only checks unit_flag and does not set unit_flag, so unit_flag remains 0. store uses Final and must wait for unit_flag to be set to 1 before reading, but unit_flag is never set to 1, so Fixpipe keeps waiting → hang.

### Error Case 2: Precision Issue Caused by store Not Configuring phase

```python
for ki in pl.range(0, K_SQ, TILE_SQ):
    ...
    if ki == 0:
        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
    else:
        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
pl.store(out, ac, [0, 0])
```

**Symptom**: Precision issue.

**Cause**:

- **From the software synchronization perspective**: store does not configure phase, so the framework automatically inserts Fixpipe synchronization. However, matmul configures phase, so the matmul computation pipeline synchronization is not automatically inserted. The two synchronization mechanisms do not match.
- **From the hardware unit_flag perspective**: store does not configure phase, so it is not affected by the hardware unit_flag value, and Fixpipe does not wait for unit_flag.

In both cases above, Fixpipe does not strictly wait for the Matmul computation to complete before moving L0C data, resulting in reading incomplete data.

### Error Case 3: matmul Hangs After store(Final) in a Loop

```python
for ki in pl.range(0, K_SQ, TILE_SQ):
    ...
    pl.matmul(ac, al, br, phase=pl.AccPhase.Final)
    pl.store(out, ac, [0, 0], phase=pl.STPhase.Partial)
```

**Symptom**: Hang.

**Cause**:

- First loop iteration: matmul(Final) sets unit_flag to 1, and store(Partial) can move the L0C data out, but it does not change the value of unit_flag (still 1).
- Second loop iteration: Because the same L0C memory is shared, matmul waits for unit_flag to change to 0, but unit_flag remains 1 → hang.
