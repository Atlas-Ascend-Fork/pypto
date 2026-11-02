# pypto_pro.language.make_ptr

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:17:22.897Z pushedAt=2026-10-28T11:57:29.131Z nl=2 -->

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

Obtains the address from an existing Tensor, or creates a new element type view for an existing Ptr. The returned Ptr shares the address with the source object. When dtype is specified, the returned Ptr interprets the data according to that element type.

It is commonly used to interpret a workspace allocated at byte granularity as a wider data type, and then wrap it into a Tensor view that can be loaded or stored using [pypto_pro.language.make_tensor](make_tensor.md). addptr offsets the pointer start by elements and preserves the element type; make_ptr preserves the pointer start and sets the element type.

## Prototype

```python
pypto_pro.language.make_ptr(ptr: Union[Tensor, Ptr], dtype: Optional[DataType] = None) -> Ptr
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| ptr | Input | Object to be converted, of the Tensor or Ptr type. When a Tensor is passed, its address is obtained; when a Ptr is passed, the original address is reused. This API does not allocate memory or copy data. |
| dtype | Input | Target element type, of the [DataType](../basic_data_structures/DataType.md) type, optional. When passed, the returned Ptr interprets the original data according to this data type; when omitted, the element type of the source object is preserved. Specifying dtype does not change the original data. The caller must ensure that the new type meets the address alignment and memory range requirements. |

## Constraints

None

## Return Value

Returns a Ptr that has the same address as the source Tensor or source pointer and an element type of dtype; when dtype is omitted, the element type of the source object is preserved.

## Examples

### Reinterpret a DT_UINT8 Pointer as a DT_FP16 Pointer

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def make_ptr_kernel(
    workspace: pl.Ptr[pl.DT_UINT8],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    fp16_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP16)
    ws_buf = pl.make_tensor(fp16_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```

### Preserve the Data Type of the Source Pointer

```python
same = pl.make_ptr(ptr)
```

### Extracting a Raw Pointer from a Tensor

```python
ptr = pl.make_ptr(tensor)
```
