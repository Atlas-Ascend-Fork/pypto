# vf.arange

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

Generates an index sequence from the start value start, which is used to construct an index vector. The generation direction is selected via index_order:

- pypto_pro.language.IndexOrder.INCREASE_ORDER (default): Ascending, with the passed-in start as the starting value.
- pypto_pro.language.IndexOrder.DECREASE_ORDER: Descending, with the passed-in start as the final value.

## Prototype

```python
arange(start, dtype: DType, *, index_order: Optional[IndexOrder] = None) -> dst
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| start | Input| Start value of the sequence (integer scalar or expression).|
| dtype | Input| Data type of the destination [reg_tensor](../basic_data_structures/reg_tensor.md) (for example, pypto_pro.language.DT_UINT32 or pypto_pro.language.DT_INT32). The register data type cannot be inferred from the scalar source and must be explicitly specified.|
| index_order | Input| Optional keyword argument. Generation direction. pypto_pro.language.IndexOrder.INCREASE_ORDER (default, ascending) or pypto_pro.language.IndexOrder.DECREASE_ORDER (descending). The step size per step is fixed at ±1; this is a hardware characteristic. If a step other than 1 is needed (for example, start + i*step), a vf.muls can be appended after vf.arange to scale the entire result.|

## Constraints

- Data type constraints:

  The supported data types for dst are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32, and DT_INT64.

## Return Value

Returns the destination dst, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). For details about the supported data types, see [Constraints](#constraints).

## Examples

### Ascending Sequence

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idxC = vf.arange(0, dtype=pl.DT_UINT32)
    vf.store_align(dst_tile, idxC, preg)

@pl.jit()
def example_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    t_out_grp = pl.make_tile_group(type=tu, addrs=0x0, mutex_ids=[0])
    t_out = t_out_grp.current()
    with pl.section_vector():
        example_vf(t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](out)
    torch.npu.synchronize()
    expected = torch.arange(64, dtype=torch.int32).unsqueeze(0).to(device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### Descending Sequence

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_dec(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idxC = vf.arange(63, index_order=pl.IndexOrder.DECREASE_ORDER, dtype=pl.DT_UINT32)
    vf.store_align(dst_tile, idxC, preg)

@pl.jit()
def example_kernel_dec(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    t_out_grp = pl.make_tile_group(type=tu, addrs=0x0, mutex_ids=[0])
    t_out = t_out_grp.current()
    with pl.section_vector():
        example_vf_dec(t_out)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel_dec[None, core_nums](out)
    torch.npu.synchronize()
    expected = torch.arange(126, 62, -1, dtype=torch.int32).unsqueeze(0).to(device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```

### INT64 Data Type Example

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.arange(0, dtype=pl.DT_INT64)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_int64(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
):
    tf = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_int64(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int64():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-100, 100, [1, 32], device=device, dtype=torch.int64)
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.arange(32, dtype=torch.int64, device=device).unsqueeze(0), rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
