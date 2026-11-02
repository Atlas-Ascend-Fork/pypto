# QuantIndexerProlog Operator Performance Optimization Case

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T02:18:35.696Z pushedAt=2026-10-27T09:16:59.064Z nl=2 -->

## Task and Objective

The DeepSeekV3.2-Exp network introduces the Indexer module. The first half of the computation in this module is referred to as IndexerProlog computation, and the computation flow of its quantized version is as follows:

![](../../../figures/tensor/docs_models_deepseek-v3-2-exp_figures_IndexerPrologQuant.png)

This operator exhibits the following characteristics:

- The computation consists of three parts: Indexer Q, Indexer Cache, and Indexer Weight. These three parts are independent of each other, and each part is executed serially.
- Among the three independent computation flows, Indexer Q has the longest execution duration, which can mask Indexer Cache and Indexer Weight.
- In a typical scenario (Batch = 4, MTP1, KV Cache length 64k), the computation workload is relatively small and does not fully occupy all cores for computation. The performance bottleneck lies in data movement.

## Analyzing Key Bottlenecks

After the precision was verified, the initial performance was obtained, also referred to as the out-of-the-box performance, which is as follows:

![](../../../figures/tensor/pre_optimization_state.png)

From the out-of-the-box performance swimlane diagram, the following performance optimization points can be observed:

- Vector tasks are both numerous and sparse, with a large number of scheduling bubbles in the execution subgraph. The issue manifests as the dequantization and RoPE computations failing to be merged into the same subgraph, which increases the number of tasks, adds to the scheduling burden, and creates scheduling gaps. The root cause is an unreasonable TileShape configuration.
- Cube computation is time-consuming. Adjust TileShape to improve Cube performance.
- L1 Reuse is not yet enabled, and some subgraphs that can be merged remain unmerged, which increases the number of tasks and causes a large amount of redundant data movement for the right cube. In typical scenarios, the operator is usually memory-bound; reducing the amount of data movement can improve performance.

## Main Tuning Process

- Tile block adjustment: After observation, in the dequantization and RoPE computation segments, there are many tasks that are not merged into a single isomorphic subgraph. If the expected Vector computations can be concentrated into one isomorphic subgraph, redundant data movement can be eliminated. By adjusting the Tile blocks in the related computations to keep them consistent, the pass will place the related computations into the same isomorphic subgraph. After this optimization, the performance improved to 76 us, as shown in the following swimlane diagram.

    ![](../../../figures/tensor/vec_optimization_swimlane.png)

- Cube Tile block adjustment: The initial Tile blocks are \([128, 128], [128, 128], [128, 128]\), which typically yield decent performance. However, since m = 8 in this operator's implementation, more suitable Tile blocks can be set to achieve better performance. For example, the m axis can be cut to 16, the k axis to 512/1024, and the n axis to 64/32. After this optimization, the performance reached 56 us, as shown in the following swimlane diagram:

    ![](../../../figures/tensor/cube_optimization.png)

- The excessive number of Cube tasks for Q computation led to repeated loading of the right cube. Therefore, L1Reuse was enabled to merge tasks and reduce redundant data movement. After optimization, the performance reached 49 us. The swimlane diagram is as follows:

    ![](../../../figures/tensor/optimized_swimlane.png)

## Obtaining the Complete Sample

In a typical scenario (Batch=4, MTP1, Kv Cache length 64k), users can write corresponding test scripts to execute different use cases of the QuantIndexerProlog operator as needed.
