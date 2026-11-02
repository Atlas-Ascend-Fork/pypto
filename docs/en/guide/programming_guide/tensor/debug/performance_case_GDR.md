# GatedDeltaRule Operator Performance Optimization Case

<!-- md-trans-meta sourceCommit=9de5cd1f921ab65753457585704a01ca5ed70c40 translatedAt=2026-10-27T02:19:10.507Z pushedAt=2026-10-28T08:51:42.857Z nl=2 -->

## Task and Objective

With its powerful attention mechanism, the Transformer delivers exceptional sequence modeling capability and has achieved breakthroughs in the field of large language models. However, the computational complexity of the self-attention module grows quadratically with sequence length, posing significant computational challenges during both training and inference. To mitigate this issue, researchers have proposed linear Transformer alternatives that replace the conventional softmax attention with kernelized dot-product attention, reformulating it as a linear RNN with cube states and substantially reducing the computational requirements in both training and inference phases. The paper *Gated Delta Networks: Improving Mamba2 with Delta Rule* introduces the Gated DeltaNet architecture, which combines a gating mechanism with the Delta update rule to enhance the performance of linear Transformers in long-sequence modeling and information retrieval tasks.

Figure 1 Gated DeltaNet model architecture
![figure caption](../../../figures/tensor/gdr_1.1.svg)

## Operator Framework

```py
@pypto.frontend.jit
def chunk_gated_delta_rule():
    # loop b in parallel
    for b_idx in pypto.loop(B):
        # loop n in parallel
        for nv_idx in pypto.loop(Nv):
            # loop s in serial
            for s_idx in pypto.loop(0, S, L):
                # view
                ...

                # compute
                # The computation logic before recurrent_loop can be executed in parallel
                ...
                # The computation logic of recurrent_loop must be serial because the output state is updated chunk by chunk
                ...

                # assemble
                ...
```

## Tuning Process

### PyPTO Config Tuning

PyPTO provides various Config options so that users can perform in-depth custom optimization for their specific fused operator implementations, such as Pass graph merge/split strategy adjustment and Runtime task scheduling strategy. Among them, the configuration most relevant to GDN operator performance tuning is stitch_function_max_num in runtime_options. Dynamic stitch is a key technology of PyPTO's MPMD architecture. It dynamically determines the control flow to be executed based on runtime inputs through AICPU, dynamically computes dependencies, and stitches tasks together for dispatch. The stitch_function_max_num configuration represents the amount of computation tasks that the ctrlflow aicpu controls at machine runtime for the first device task submitted to the schedule aicpu for processing. This value is used to control the startup overhead of the device machine, so that the computation of the ctrlflow aicpu and the schedule aicpu can overlap as soon as possible.
The following swimlane diagrams are all for the data scenario of B=2, T=8192, H=4, D=128. As shown in Figure 2 and Figure 3, they are respectively the swimlane diagram with stitch_function_max_num set to 32 and the swimlane diagram with stitch_function_max_num set to 128. The larger the value, the more fully tasks can be parallelized, resulting in better performance, but the workspace also increases, which requires a trade-off.

Figure 2 Swimlane diagram with stitch_function_max_num set to 32
![figure caption](../../../figures/tensor/gdr_3.1.png)

Figure 3 Swimlane diagram with stitch_function_max_num set to 128
![figure caption](../../../figures/tensor/gdr_3.2.png)

### Ascend-Friendly Tile Shape Size Tuning

In PyPTO, all computations are based on Tile (hardware-aware data blocks), fully leveraging the hardware's parallel computing capability and memory hierarchy. Tiles can be stored in AICore private caches (such as UB and L1), thereby significantly improving data access efficiency. For each OP operation, its tile_shape size can be flexibly set to optimize computation load balancing and memory bandwidth utilization, and to maximize the efficiency of hardware resource usage, so as to fully utilize the UB capacity and fit the Ascend architecture characteristics. Triton operators developed based on GPU usually set the block size to 64, in which case the Tile block shape participating in computation is [64, 128]. When dtype is FP32, the data size can be calculated as 32KB. When the UB of an Ascend NPU is typically 192KB, the appropriate Tile block size is selected as 16-64KB. After adjusting the chunksize to 128, the data size can be increased to 64KB, which better conforms to the Ascend operator development strategy of moving a larger data block at a time, greatly reducing the performance overhead caused by data movement and fully leveraging the hardware's parallel computing capability. In addition, due to the chunk loop update algorithm characteristic of the GDN operator, processing a larger chunksize of data at a time can reduce the number of serial loop updates, alleviating the performance problem that serial computation logic is difficult to fully parallelize and fully utilize cores under PyPTO's existing MPMD scheduling strategy.
As shown in Figure 4, compared with Figure 3, this is the swimlane diagram with chunk_size optimized from 64 to 128 and TileShape optimized from [64, 128] to [128, 128]. The Tile block data size is more compatible with the Ascend NPU hardware storage size, which can reduce the number of Loop iterations, the number of root functions, and the number of tasks. Under the premise that the tasks bound and dispatched by stitch remain unchanged (stitch_function_max_num=128), the number of stitches is reduced, which essentially reduces the data movement overhead and improves computation efficiency.

Figure 4 Swimlane diagram with chunk_size set to 128 and TileShape set to [128, 128]
![figure caption](../../../figures/tensor/gdr_3.3.png)

### PyPTO Loop Mode and Graph Stitching and Unrolling Optimization

#### Adjusting Dynamic Loop to Static Loop

PyPTO adopts the PTO (Parallel Tensor/Tile Operation) programming paradigm, with a tile-based programming model as its core design philosophy. Through multi-level compute graph representation, it compiles the AI model built by users via APIs from high-level tensor compute graphs down to hardware instructions, ultimately generating code that can be efficiently executed on the target platform. The device side then automatically schedules execution in MPMD (Multiple Program Multiple Data) mode. Understanding PyPTO loops and the graph stitching and splitting logic is a further step toward deeper comprehension of its frontend representation. First, the loops in PyPTO are primarily designed to handle dynamic shapes. In a full-network scenario, parameters such as batch_size and seq_length are often dynamic, while the tile size is typically fixed. The PyPTO framework translates pre-marked dynamic axes into CCE code using expressions composed of SymbolicScalar during compilation, and then resolves the specific sizes of these expressions at the Machine layer. This enables dynamic handling of on-device execution states for different shapes, significantly reducing the repetitive overhead of compilation graph construction and allowing different dynamic shape inputs to share the same frontend representation and graph IR. In the GDN operator, the inverse module performs row-by-row computation and update on tile blocks. When tail blocks are not considered (they can also be padded), the number of row-by-row loop iterations is fixed. However, the actual number of PyPTO loop iterations occupies the root function, which is the number of tasks that can be bound and dispatched in a single stitch. Therefore, in terms of programming paradigm, the inverse module is better suited to using Python's static loop approach. While this sacrifices some compilation performance, it greatly reduces the runtime overhead incurred by the inverse module—which requires extensive repetitive computation—when dynamic loops are used.

```py
# row_num is a static value
# Original dynamic loop mode
for i in pypto.loop(2, row_num, 1):
    # Update the inverse row by row

# Change to static loop mode
for i in range(2, row_num, 1):
    # Update the inverse row by row
```

#### Graph Stitching and Splitting Optimization

The computation graph of PyPTO consists of Tensor data nodes and Operation nodes. After layer-by-layer Pass optimization, the Tensor Graph computation graph defined by the user's frontend representation is finally converted into an Execution Graph, which is then translated into CCE executable code. The Execute Graph integrates the computation subgraph information, including their respective dependency relationships and scheduling information, and defines the specific Operation combinations of Tiles on AIC/AIV. Different graph stitching and splitting optimization methods have a great impact on performance. Pass usually provides generalized optimization strategies to help users obtain better performance. However, if users need to control the specific computation flow of Tile blocks, users need to leverage the DFX capability of the computation graph to visually check whether it meets expectations. As shown in Figure 5, this is the initial out-of-the-box swimlane diagram for B=1, T=128, H=1, where many extremely short-duration fragments and abnormally long-running execution graphs are found. After inspecting and comparing the computation flow of the computation graph, it is found that the PyPTO framework itself had certain issues with graph stitching and splitting, causing modules such as the inverse module to fail to be correctly stitched and split according to the frontend representation. After subsequent communication and discussion with the framework developers, the issue can be resolved. As shown in Figure 6, this is the swimlane diagram after manual tuning of graph stitching and splitting for B=1, T=128, H=1. At this point, the computation logic graph is highly consistent with the frontend representation. For example, the inverse module is a homogeneous subgraph of eight 16\*16 Tile blocks updated row by row in parallel. PyPTO provides users with a certain degree of DFX visualization capability, which helps users understand the specific runtime computation flow more easily and intuitively, thereby discovering and resolving issues.

Figure 5 Initial out-of-the-box swimlane diagram for B=1, T=128, H=1
![figure caption](../../../figures/tensor/gdr_3.4.png)

Figure 6 Swimlane diagram after manual tuning of graph stitching and splitting for B=1, T=128, H=1
![figure caption](../../../figures/tensor/gdr_3.5.png)

#### unroll_list Optimization

After understanding the runtime concepts such as root function and stitch, we can observe from the swimlane diagram that the GDN operator is composed of parallel computation logic plus serial computation logic. Due to the current Stitch strategy and scheduling strategy of the PyPTO framework, when S is large, the per-chunk state update is difficult to fully utilize the cores. PyPTO provides the unroll_list capability to merge the innermost Loop graphs. When the unroll count is n, the loop step becomes step*n, and each iteration executes the loop body n times, thereby improving the parallel computation capability of the outer BN. In essence, this also reduces the number of Loop iterations, the number of root functions, and the number of tasks. Under the premise that the number of tasks dispatched under stitch binding remains unchanged (stitch_function_max_num=128), the number of stitches is reduced. As shown in Figure 7, compared with Figure 4, this is the swimlane diagram when unroll_list=[16] is set in Loop S.

```py
# Original
for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx"):

# Set unroll_list=[16]
for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16]):
```

Figure 7 Swimlane diagram with unroll_list=[16]
![Figure description](../../../figures/tensor/gdr_3.6.png)

### DFX-based In-Depth Performance Tuning

As shown in Figure 9, in the initial out-of-the-box implementation of GDN, for the inversion of the [128, 128] tile block, we first perform the inversion algorithm on eight [16, 16] blocks. This approach offers strong parallel computation capability in small-shape scenarios. However, as shown in Figure 10, under large data volumes (when the AIV cores are already fully utilized), the computation efficiency is not high. Based on the DFX capability provided by the computation graph and swimlane diagram, we proposed an inverse tail-axis merge optimization scheme that concatenates the eight [16, 16] blocks along the tail axis into two [16, 64] blocks for computation, so as to reduce the data movement overhead and improve computation efficiency. However, the resulting performance gain did not meet expectations. As shown in Figure 8, the time was only optimized from 8\*22 us to 2\*62 us, which did not match the expected gain. After inspecting the computation flow in the computation graph, we found that it contained considerable copy-in and copy-out overhead, as shown by the green and red nodes in Figure 9.

Figure 8 Swimlane diagram of the initial puncture scheme for inverse tail-axis merge optimization
![Figure description](../../../figures/tensor/gdr_3.7.png)

Figure 9 Computation graph of the initial puncture scheme for inverse tail-axis merge optimization (green nodes represent copy-in, red nodes represent copy-out)
![Figure description](../../../figures/tensor/gdr_3.8.png)

By comparing the computation flow in the computation graph with the frontend code, we found that the issue was caused by the row-by-row updated cube being frequently moved in and out of UB. However, it was expected that this cube could reside permanently in memory, which would enable deeper optimization. As shown in Figure 10, in the optimized computation graph, data is copied in only at the far left of the computation flow and copied out at the far right, which aligns with expectations.

Figure 10 Computation graph of the puncture scheme for inverse tail-axis merge optimization
![Figure description](../../../figures/tensor/gdr_3.9.png)

Finally, the tail-axis merge scheme is optimized by concatenating into a single [16, 128] block for row-by-row inversion. The resulting swimlane diagram of the final puncture scheme for inverse tail-axis merge optimization is shown in Figure 11. The inversion module time was optimized from 8\*28 us to 49 us.

Figure 11 Swimlane diagram of the final puncture scheme for inverse tail-axis merge optimization
![Figure description](../../../figures/tensor/gdr_3.10.png)
