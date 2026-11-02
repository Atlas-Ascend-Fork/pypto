# pypto_pro.language.PipeType

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T12:46:28.595Z pushedAt=2026-10-28T11:57:28.633Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Enumeration of hardware execution unit (pipeline) types, marking which pipeline an operation runs on. It is the source of values for the set_pipe/wait_pipe/pipe parameters in synchronization control ([sync_src](../synchronization/sync_src.md), [sync_dst](../synchronization/sync_dst.md), [mutex_lock](../synchronization/mutex_lock.md), and so on).

The AI processor has multiple parallel pipelines, each responsible for data movement and computation at different stages. Correctly understanding the responsibility of each pipeline is the foundation of writing proper synchronization control.

## Prototype

```python
PYPTO_DECLARE_ENUM(
    PipeType,
    MTE1,
    MTE2,
    MTE3,
    M,
    V,
    S,
    FIX,
    ALL
)
```

## Parameters

| Parameter Value | Description |
|---|---|
| MTE1 | Data transfer pipeline 1, responsible for data movement from L1 Buffer to L0A Buffer/L0B Buffer/BiasTable Buffer/L0A_MX Buffer/L0B_MX Buffer. |
| MTE2 | Data transfer pipeline 2, responsible for load-in from GM to L1 Buffer/UB. |
| MTE3 | Data transfer pipeline 3, responsible for store-out from UB to GM and move from UB to L1 Buffer. |
| M | Matrix computation pipeline, responsible for matmul computations such as Cube and MAD. |
| V | Vector computation pipeline, responsible for vector operations such as element-wise, reduce, and cast. |
| S | Scalar pipeline, responsible for scalar operations such as getval and setval. |
| FIX | Fixpipe pipeline, responsible for reading out accumulator results (L0C Buffer → UB/GM/L1 Buffer) and in-line quantization/dequantization (L0C Buffer → UB/GM). |
| ALL | Indicates all pipelines of this AI Core, including V, M, MTE1, MTE2, MTE3, and FIX. |

## Constraints

### Data Movement Scenarios

The pipelines for [load](../memory_data_movement/load.md), [move](../memory_data_movement/move.md), and [store](../memory_data_movement/store.md) are automatically determined by the source/destination memory space:

| Operations | Source → Destination | Pipeline |
|---|---|---|
| load | GM → L1 Buffer/UB | MTE2 |
| store | UB → GM | MTE3 |
| store | L0C Buffer → GM | FIX |
| move | L1 Buffer → L0A Buffer/L0B Buffer/BiasTable Buffer/L0A_MX Buffer/L0B_MX Buffer | MTE1 |
| move | L1 Buffer → Fixpipe Buffer | FIX |
| move | L1 Buffer → UB | V |
| move | L0C Buffer → UB | FIX |
| move | UB → UB | V |
| move | UB → L1 Buffer | MTE3 |
| move | Others | V |

### Pipeline Synchronization Scenarios

Typical synchronization patterns:

| Scenario | set_pipe | wait_pipe | Description |
|---|---|---|---|
| V can compute only after load | MTE2 | V | Ensures that the GM→UB data movement is complete. |
| MTE3 can store only after computation | V | MTE3 | Ensures that the vector computation is complete. |
| MTE2 can load only after store | MTE3 | MTE2 | Ensures that the UB→GM data movement is complete before new data is moved in. |
| M can compute only after the L1 Buffer→L0A Buffer data movement | MTE1 | M | Ensures that the matrix operands are in place. |
| FIX can read the result only after matmul | M | FIX | Ensures that the matrix computation is complete. |
| Scalar read/write | MTE2/MTE3 | S | getval/setval go through the scalar pipeline. |

## Examples

### Pipeline Synchronization

```python
import pypto_pro.language as pl
# After load synchronization: MTE2 is set, and V waits.
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

# After compute synchronization: V is set, and MTE3 waits.
pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
```
