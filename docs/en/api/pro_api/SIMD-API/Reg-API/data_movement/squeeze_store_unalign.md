# vf.squeeze_store_unalign

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

Unaligned store, which writes variable-length vector data into a Tile. Its basic functionality is the same as the reg_tensor mode of vf.store_unalign, but the parameters and constraints differ.

- In the reg_tensor mode of vf.store_unalign, two additional parameters, stride and post_update, are required for configuration, whereas vf.squeeze_store_unalign does not have these two parameters.

- For details about the differences in constraints, see the constraint description sections of vf.store_unalign and vf.squeeze_store_unalign.

## Prototype

```python
squeeze_store_unalign(tile, src, align_reg, /)
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| tile | Output| Destination operand, Tile address.|
| src | Input| Source operand, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64.|
| align_reg | Input| alignment tracker register (created by vf.unalign_reg_for_store()).|

## Constraints

- Before calling this API, call vf.squeeze to write data to the AR register.

- This API must be used together with vf.squeeze_store_unalign_post and called before vf.squeeze_store_unalign_post.

- vf.squeeze and vf.squeeze_store_unalign must be used alternately. Otherwise, a hardware hang may occur.

## Return Value

None

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src_tile, 0)
    vf.clear_spr()
    reg_sq = vf.squeeze(reg_src, preg, gather_mode=pl.SqueezeMode.STORE_REG)
    align_reg = vf.unalign_reg_for_store()
    vf.squeeze_store_unalign(dst_tile, reg_sq, align_reg)
    vf.squeeze_store_unalign_post(dst_tile, align_reg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.zeros([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
