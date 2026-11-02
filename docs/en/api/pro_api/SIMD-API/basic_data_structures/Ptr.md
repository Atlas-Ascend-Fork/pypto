# pypto_pro.language.Ptr

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:46:24.777Z pushedAt=2026-10-28T11:57:28.627Z nl=2 -->

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

A GM address type annotation that points to the specified element type.

pypto_pro.language.Ptr is mainly used to:

1. Declare GM raw pointer parameters in Kernel function signatures
2. Create pointer views of different element types with [pypto_pro.language.make_ptr](../resource_management/make_ptr.md)
3. Perform pointer offset with [pypto_pro.language.addptr](../resource_management/addptr.md)
4. Construct a Tensor view from a raw pointer with [pypto_pro.language.make_tensor](../resource_management/make_tensor.md)

## Prototype

```python
pypto_pro.language.Ptr[dtype]
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dtype | Input | Data type of the elements pointed to by the pointer, an enumeration value of [pypto_pro.language.DataType](DataType.md). Common values: pypto_pro.language.DT_FP16, pypto_pro.language.DT_FP32, and pypto_pro.language.DT_INT8. |

## Constraints

None

## Return Value

None

## Examples

In this example, pypto_pro.language.Ptr is used in the Kernel signature to declare a workspace pointer. pypto_pro.language.addptr offsets the pointer by 64 × 128 FP16 elements, and pypto_pro.language.make_tensor creates a Tensor view with the shape [64, 128]. The example moves the data from the input Tensor into UB, multiplies each element by 2, writes the result to the workspace, then moves the data from the workspace into UB and writes it to the output Tensor.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```
