# Stack Overflow Error During Operator Compilation

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:18:26.713Z pushedAt=2026-10-27T03:01:44.588Z nl=2 -->

## Symptoms

During operator compilation, an error message similar to the following appears:

```text
error: stack frame size (*****) exceeds limit (32768) in function '*****'
```

The following is an example in the console log:

```text
error: stack frame size (47928) exceeds limit (32768) in function 'TENSOR_nLoop_Unroll1_PATH0_6_0_4503599627370496'
error: stack frame size (47928) exceeds limit (32768) in function 'TENSOR_nLoop_Unroll1_PATH0_6_0_4503599627370496'
2 errors generated.
terminate called after throwing an instance of 'npu::tile_fwk::Error'
  what():  ASSERTION FAILED: ret == 0
CompileCCE failed. errCode = 256, cce file: output/output_20251111_175724_806073/kernel_aicore/TENSOR_nLoop_Unroll1_PATH0_6_17699850674043372772_0_aic.cpp
```

## Possible Causes

Due to hardware limitations, the stack space of the Scalar Processing Unit (SPU) within an Ascend AI processor core is capped at 32 KB. Therefore, during operator compilation, the underlying BiSheng Compiler analyzes and verifies the stack usage of the operator's kernel function. If the function implementation is complex—for example, due to a large number of variables or long variable lifetimes—the analysis may be affected. If the final analysis result exceeds the 32 KB limit, the BiSheng Compiler will intercept and report an error.

Under the PTO programming model, the main reasons for complex operator kernel function implementations are as follows:

- The subgraph is large in scale, resulting in an excessive number of variables.
- The subgraph computation logic is complex, resulting in long variable lifetimes.

## Solution

Based on the possible causes described above, you can take the following measures:

- Adjust the tensor tiling strategy by using larger tile sizes. By increasing the workload per tile while keeping the total data volume unchanged, you can reduce the number of computation steps in the subgraph, thereby effectively reducing its size. The related configuration APIs are [pypto.set_vec_tile_shapes](../../../api/tensor_api/config/pypto-set_vec_tile_shapes.md) and [pypto.set_cube_tile_shapes](../../../api/tensor_api/config/pypto-set_cube_tile_shapes.md).
- For cube multiplication (MATMUL) scenarios, it is recommended that users perform multi-core splitting along the K-axis.
- Modify the `cycle_upper_bound` option that controls the subgraph size to limit the maximum size of a single subgraph within a specified range. The related configuration API is [pypto.set_pass_options](../../../api/tensor_api/config/pypto-set_pass_options.md).
