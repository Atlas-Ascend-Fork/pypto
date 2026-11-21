# Introduction

A warp is the basic unit of SIMT thread execution and scheduling. Threads in a thread block are divided into multiple warps based on linear thread indices. Currently, a warp contains 32 threads. Threads in the same warp execute the same instruction stream. Each thread has an independent register and lane ID, and can process different data addresses and branch paths.

Warp functions are used for lightweight thread collaboration within the same warp, providing capabilities such as lane information query, conditional vote, register data exchange, and warp-level reduction. Compared with data transfer through the Unified Buffer (UB), the warp shuffle function can directly exchange scalar values held by lanes. This is suitable for low-overhead small-scale collaboration.

The following table describes the related basic concepts.

| Concept| Description|
|---|---|
| Warp | A group of threads in SIMT execution. Currently, a warp contains 32 threads.|
| Lane | An execution position of a single thread in a warp.|
| Lane ID | The ID of a thread in the current warp. The value range is [0, 31].|
| Active thread| A thread that actually participates in execution at the current instruction position. It is also called an active lane. The active status is related to the instruction position, and is not a fixed attribute of a thread.|
| Branch divergence| Threads in the same warp enter different branch paths. The hardware executes each path in sequence and temporarily masks the lanes that do not belong to the current path.|
| Logical subgroup| An independent exchange group obtained by dividing a warp using the width of the shuffle function.|

**Active threads** are the threads that actually execute instructions at the **current instruction position**. The active status changes with the program execution position. A lane may be temporarily inactive in a branch path, but becomes active again after entering the subsequent common path. Therefore, "inactive" does not mean that the thread has ended. It only indicates that the lane does not execute instructions on the current path.

**Branch divergence** occurs when lanes in the same warp enter different branches based on their own data. The hardware first executes one of the paths and masks the lanes that do not belong to that path. Then, it executes the other paths. After all paths are executed, the lanes converge again on the common path. The following uses a warp with 32 lanes as an example. The first 16 lanes and the last 16 lanes enter different branches.

```python
lane = pl.simt.lane_id()
if lane < 16:
    lower_mask = pl.simt.warp_active_mask()
else:
    upper_mask = pl.simt.warp_active_mask()
merged_mask = pl.simt.warp_active_mask()
```

The active lanes observed at each call site are listed below.

| Call Site| Active Lanes Executing Instructions| 32-Bit Active Mask (Bits 31 to 0)|
|---|---|---|
| lower_mask | Lanes 0 to 15| 00000000 00000000 11111111 11111111 |
| upper_mask | Lanes 16 to 31| 11111111 11111111 00000000 00000000 |
| merged_mask | Lanes 0 to 31| 11111111 11111111 11111111 11111111 |

Although lower_mask and upper_mask are in the same warp, they are in different branch paths. Therefore, different active lane sets are obtained. After the branch ends, the 32 lanes converge again at merged_mask.
