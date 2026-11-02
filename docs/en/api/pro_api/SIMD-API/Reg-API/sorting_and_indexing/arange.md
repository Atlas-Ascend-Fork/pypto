# vf.arange

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:08:24.799Z pushedAt=2026-10-28T11:57:29.081Z nl=2 -->

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

Generates an index sequence from the start value start, which is used to construct an index vector. The generation direction is selected through index_order:

- pypto_pro.language.IndexOrder.INCREASE_ORDER (default): Incrementing, with the passed start as the start value.
- pypto_pro.language.IndexOrder.DECREASE_ORDER: Decrementing, with the passed start as the final value.

## Prototype

```python
arange(start, dtype: DType, index_order: Optional[IndexOrder] = None) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| start | Input | Sequence start value (an integer scalar or expression). |
| dtype | Input | Data type of the target [reg_tensor](../basic_data_structures/reg_tensor.md) (for example, pypto_pro.language.DT_UINT32, pypto_pro.language.DT_INT32). Because the register data type cannot be inferred from a scalar source, it must be specified explicitly. |
| index_order | Input | Optional keyword argument that specifies the generation direction: pypto_pro.language.IndexOrder.INCREASE_ORDER (default, incrementing) or pypto_pro.language.IndexOrder.DECREASE_ORDER (decrementing). The step size of each step is fixed at ±1, which is a hardware characteristic. If a step size other than 1 is required (for example, start + i*step), append a vf.muls after vf.arange to scale the result as a whole. |

## Constraints

- Data type constraints:

  Supported dst data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32.

## Return Value

Returns the dst target [reg_tensor](../basic_data_structures/reg_tensor.md), which stores the generated sequence. For supported data types, see [Constraints](#constraints).

## Examples

### Increasing Sequence

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

### Decreasing Sequence

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
