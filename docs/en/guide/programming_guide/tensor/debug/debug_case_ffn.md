# ffn_shared_expert_quant Operator NPU On-Board Debugging Case

<!-- md-trans-meta sourceCommit=6c3d23fb55eb8bacdad9a9cfbcb9208870fc071b translatedAt=2026-10-27T02:15:33.517Z pushedAt=2026-10-27T09:31:39.521Z nl=2 -->

## Task and Objective

The ffn_shared_expert_quant operator corresponds to the computation logic of the MoE shared expert in the GLM4.5 network. It includes symmetric_quantization_per_token, matmul, dequant_dynamic, and swiglu, and is used to perform quantized forward propagation computation for a single shared expert. By reusing the same set of weight parameters across different tasks or data flows, it learns general feature representations while reducing the total number of model parameters.

The following uses this operator to introduce the general steps of functional debugging.

## Locating the Issue

Assume that the ffn_shared_expert_quant operator test case fails and reports an error when running on the NPU board. In this case, debugging is required to locate the issue.

First, you can locate the issue by viewing the Tensor Graph. The main steps are as follows:

1. Enable the debug mode by following the steps described in [Enabling Debug Mode](debug.md), and then re-run the case to obtain the compute graph of the ffn_shared_expert_quant operator.
2. As described in [Viewing the Compute Graph](debug.md), use the PyPTO Toolkit visualization tool to open the compute graph file at the Tensor Graph stage, for example, Before\_004\_ExpandFunction\_TENSOR\_share\_loop\_idx\_Unroll1\_PATH0\_4.json:

    ![](../../../figures/tensor/fnn_0.png)

3. The actual operator code calls the Matmul API for the operation. However, as shown in the preceding figure, there is no Matmul operation node, and all subsequent tensor and Operation nodes failed to load, indicating an obvious anomaly.

    Check whether each Matmul operation in the operator code is used correctly. It was found that the A/B cube positions in the first Matmul input were incorrect, causing the reduce axes of the A/B cubes to violate the equality constraint. The error exists in:

    ```python
    up_proj = pypto.matmul(w13, hidden_states_quant, pypto.DT_INT32)
    ```

In addition to debugging with the compute graph, the issue can also be located using the internal DFX verification mechanism.

The following shows the ERROR information displayed when the ffn_shared_expert_quant operator test case failed:

```text
ERROR:root:Record function share_expert_moe_main failed: ASSERTION FAILED: kSizeA == kSizeB
Matrix K dimension mismatch, kSizeA: 384, kSizeB: 8
, func ConstructTensorGraph, file cube_operation_impl.cpp, line 1220
libtile_fwk_interface.so(npu::tile_fwk::Tensor npu::tile_fwk::Matrix::ConstructTensorGraph<false, false, false>(npu::tile_fwk::DataType, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&, npu::tile_fwk::Matrix::MatmulExtendParam const&)+0x25d) [0x7fe34630ad3d]
libtile_fwk_interface.so(npu::tile_fwk::Tensor npu::tile_fwk::Matrix::Matmul<false, false, false>(npu::tile_fwk::DataType, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&)+0x14e) [0x7fe34630b51e]
```

The key information "Matrix K dimension mismatch" is obtained, indicating that the error is caused by a mismatch in the K dimension of the tensor shapes passed to a Matmul operation.

## Solution

Modify the implementation code of this operator:

```python
up_proj = pypto.matmul(w13, hidden_states_quant, pypto.DT_INT32)
```

The modification result is as follows:

```python
up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_INT32)
```

Run the ffn_shared_expert_quant operator test case again. It passes successfully, and the issue is resolved.
