# pypto_pro.language.PipeType

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

Enumerates the types of hardware execution units (pipelines). It is used to mark the pipeline on which an operation is performed. It is the source of values of the set_pipe, wait_pipe, and pipe parameters in synchronization control (such as [sync_src](../synchronization/sync_src.md), [sync_dst](../synchronization/sync_dst.md), and [mutex_lock](../synchronization/mutex_lock.md)).

The AI processor has multiple parallel pipelines, each of which is responsible for data movement and computation in different phases. Correctly understanding the responsibilities of each pipeline is the basis for writing synchronization control.

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

| Value| Description|
|---|---|
| MTE1 | Data movement pipeline 1, which is responsible for data movement from the L1 Buffer to the L0A Buffer, L0B Buffer, BiasTable Buffer, L0A_MX Buffer, or L0B_MX Buffer.|
| MTE2 | Data movement pipeline 2, which is responsible for data movement from the GM to the L1 Buffer or UB.|
| MTE3 | Data movement pipeline 3, which is responsible for data movement from the UB to the GM (store) and from the UB to the L1 Buffer (move).|
| M | Matrix computation pipeline, which is responsible for matmul computations such as Cube/MAD operations.|
| V | Vector computation pipeline, which is responsible for vector operations such as element-wise, reduce, cast, quant, and dequant.|
| S | Scalar pipeline, which is responsible for scalar operations such as getval and setval.|
| FIX | Fixpipe, which is responsible for moving the accumulator results from the L0C Buffer to the GM, UB, or L1 Buffer, and performing operations such as in-line quantization/dequantization.|
| ALL | All the pipelines of the AI Core, including the V, M, MTE1, MTE2, MTE3, and FIX pipelines.|

## Constraints

### Data Movement Scenarios

The pipelines for [load](../memory_data_movement/load.md), [move](../memory_data_movement/move.md), and [store](../memory_data_movement/store.md) are automatically determined by the source and destination memory spaces.

| Operation| Source → Destination| Pipeline|
|---|---|---|
| load | GM → L1 Buffer/UB | MTE2 |
| store | UB → GM | MTE3 |
| store | L0C Buffer → GM | FIX |
| move | L1 Buffer → L0A Buffer/L0B Buffer/BiasTable Buffer/L0A_MX Buffer/L0B_MX Buffer | MTE1 |
| move | L1 Buffer → Fixpipe Buffer | FIX |
| move | L1 Buffer → UB | V |
| move | L0C Buffer → UB | FIX |
| move/insert | L0C Buffer → L1 Buffer | FIX |
| move | UB → UB | V |
| move | UB → L1 Buffer | MTE3 |
| move | Other| V |

### Pipeline Synchronization Scenarios

Typical synchronization patterns are described below.

| Scenario| set_pipe | wait_pipe | Description|
|---|---|---|---|
| V can compute only after the load operation.| MTE2 | V | Ensures that the GM-to-UB data movement is complete.|
| MTE3 can store data only after computation.| V | MTE3 | Ensures that the vector computation is complete.|
| MTE2 can load data only after the store operation.| MTE3 | MTE2 | Ensures that the UB-to-GM data movement is complete before new data is moved in.|
| M can compute only after the L1 Buffer-to-L0A Buffer data movement.| MTE1 | M | Ensures that the matrix operands are in place.|
| FIX can read the result only after the matmul operation.| M | FIX | Ensures that the matrix computation is complete.|
| Scalar read/write| MTE2/MTE3 | S | getval and setval go through the scalar pipeline.|

## Examples

### Pipeline Synchronization

```python
import pypto_pro.language as pl
# Synchronization after load: MTE2 is set, and V waits.
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

# Synchronization after computation: V is set, and MTE3 waits.
pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
```
