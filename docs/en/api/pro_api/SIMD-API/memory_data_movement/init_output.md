# pypto_pro.language.init_output

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

Initializes a specified area of a Global Memory (GM) Tensor to a scalar value. Internally, the framework first uses `TEXPANDS` on UB to fill a temporary buffer with the scalar (V pipeline), then writes the data from UB to GM using `TSTORE` (MTE3 pipeline), and automatically inserts a V→MTE3 synchronization event. This API is mainly used in preprocessing scenarios, such as batch initializing or zeroing out GM memory like workspaces before Tile buffers are allocated, eliminating the need for users to manually create and manage Tiles.

The difference between this API and [pypto_pro.language.expands](../tile_computation/math_functions/expands.md) is that expands fills on-chip Tiles (UB/L1 Buffer), requiring users to create Tiles and manage UB addresses. init_output directly initializes GM Tensors, with the framework automatically allocating temporary UB buffers, making it suitable for batch initialization of large workspace sections.

## Prototype

```python
pypto_pro.language.init_output(
    tensor: Tensor,
    *,
    offset: int = 0,
    size: int,
    value: Union[int, float] = 0,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| tensor | Output| Destination operand, of the Tensor type, stored in GM. The supported data types are DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_BF16, DT_FP16, DT_UINT32, DT_INT32, DT_FP32, DT_UINT64, and DT_INT64. The layout supports ND and NZ. The offset and size are counted based on the physical elements of the Tensor. The layout does not affect the padding semantics.|
| offset | Input| Optional, element-wise offset, which is an integer constant or a runtime integer scalar expression. It indicates the offset from the start address of the Tensor. The unit is the number of elements. The default value is `0`.<br>- A non-negative integer. Floating-point values are not supported.<br>- The sum of this parameter and size cannot exceed the total number of elements in the Tensor.<br>- Runtime scalars (such as loop variables and shape expressions) can be passed.|
| size | Input| Number of elements to be initialized. The value is an integer constant or a runtime integer scalar expression. The unit is the number of elements.<br>- A positive integer. Floating-point values are not supported.<br>- Runtime scalars can be passed.|
| value | Input| Optional, padding value. The value is an integer or floating-point constant, or a runtime scalar expression of the corresponding type. The default value is `0`.<br>- Integer or floating-point constants (such as `0` and `0.0`), scalars constructed by [pypto_pro.language.const](../../Utils-API/python_syntax_sugar/const.md), and runtime scalar expressions are supported.<br>- The type must be compatible with the tensor element type. The integer literal must be representable by the tensor data type.<br>- When the tensor is of an integer type and the value is a floating-point constant, the value is truncated towards zero and converted to an integer (the fractional part is discarded). This is consistent with the scalar conversion of other integer destination operands in the framework.|

## Constraints

- This API is a Vector pipeline (V/MTE3) operation and must be called in the Vector execution domain identified by [pypto_pro.language.section_vector](../controlflow/section_vector.md). If this API is directly called in a Kernel without an execution domain identifier, the framework considers the Kernel to be shared by the Cube and Vector, and compiles the same vector section into the Cube core (AIC) target, resulting in a compilation failure.
- This API is mainly used in preprocessing scenarios and must be used before Tile buffer allocation (such as [pypto_pro.language.make_tile_group](../resource_management/make_tile_group.md)).
- When this API is called on a single core, pipeline synchronization related to UB reuse will not be inserted into the API. In addition, when auto_mutex is enabled, the framework is not aware of the UB usage inside the API. If subsequent operations involve the use of the Unified Buffer (UB) (such as [pypto_pro.language.load](load.md)), you must set the synchronization of the MTE2 pipeline waiting for the MTE3 pipeline (`MTE3_MTE2`). Call [pypto_pro.language.system.sync_src](../synchronization/sync_src.md) and [pypto_pro.language.system.sync_dst](../synchronization/sync_dst.md) (set_pipe is MTE3 and wait_pipe is MTE2) to ensure that the UB area is reused only after the internal read of the temporary UB buffer is complete.
- When calling this API consecutively multiple times on a single core with different values, you must insert a pipeline synchronization between the API calls to make the Vector (V) pipeline wait for the MTE3 pipeline (`MTE3_V`). Because this API internally uses the same UB as an intermediate space for value initialization, consecutive calls without pipeline synchronization may cause subsequent write operations to overwrite incomplete data from the prior write, resulting in unexpected initialization outcomes in the Global Memory.
- When multiple cores call this API to initialize data in the Global Memory, the initialization may not complete simultaneously across all cores. Additionally, data dependency issues such as read-after-write, write-after-read, and write-after-write may occur between cores. **This API does not insert a full-core synchronization barrier internally. (`pipe_barrier` only applies to the current core that calls this API.) After calling this API, you must explicitly use [pypto_pro.language.system.sync_all](../synchronization/sync_all.md) (or an equivalent cross-core synchronization primitive) to ensure that the initialization results are visible to other cores (including Cube sub-cores)** before performing subsequent operations. For example, when init_output is followed by cube operations such as matmul, `system.sync_all(core_type=SyncCoreType.MIX)` needs to be called separately on the AIV and AIC sides.

## Return Value

None

## Examples

### Basic Usage

The following is a complete Kernel: Use pypto_pro.language.init_output to initialize all elements of the FP32 GM Tensor to `0.0`. When auto_mutex is enabled for the Vector Kernel, synchronization is automatically managed by [pypto_pro.language.make_tile_group](../resource_management/make_tile_group.md).

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def init_output_kernel(
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    with pl.section_vector():
        pl.init_output(out, offset=0, size=64 * 64, value=0.0)
```

### Initializing the Workspace by Block

In scenarios such as Attention, the workspace area needs to be initialized by block. The following example shows how to clear the specified offset area of the workspace block by block in a loop, with each block containing `128 * 128` elements:

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def init_dq_workspace_kernel(
    workspace: pl.Tensor[[4096 * 4096], pl.DT_FP32],
    dq_offset: pl.DT_INT32,
    init_size: pl.DT_INT32,
):
    with pl.section_vector():
        for i in pl.range(0, init_size, 128 * 128):
            pl.init_output(workspace, offset=dq_offset + i, size=128 * 128, value=0.0)
```

Other typical usage (selected):

```python
with pl.section_vector():
    # Use pl.const to construct a scalar value.
    pl.init_output(workspace, offset=dq_offset, size=128 * 128,
                   value=pl.const(0, pl.DT_FP32))

    # Initialize to a non-zero initial value (for example, negative infinity, used for mask workspace).
    pl.init_output(mask_workspace, offset=0, size=1024,
                   value=pl.const(float('-inf'), pl.DT_FP32))
```
