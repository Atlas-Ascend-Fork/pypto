# vf.store_unalign_post

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

Performs post-processing for the unaligned store by handling the remaining unaligned bytes in the unaligned register (UnalignRegForLoad). This API must be called after vf.store_unalign.

## Prototype

```python
store_unalign_post(tile, align_reg, stride, /, *, post_update: bool = False)
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| tile | Output| Destination operand, Tile address. The destination operand and the source operand must have the same data type. The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_INT64, DT_UINT64, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, and DT_FP4E1M2.|
| align_reg | Input| alignment tracker register (created by vf.unalign_reg_for_store()).|
| stride | Input| Number of stored elements or an address register. When this parameter is an integer scalar, the vstas instruction is issued (strided mode); when post_update is True, the parameter also serves as the address update step, and is valid only when post_update is True. When the parameter is an AddrReg (created by vf.create_addr_reg), the vsta instruction is issued (AddrReg mode), and the parameter must be used in pairs with the AddrReg mode (vstu) of vf.store_unalign.|
| post_update | Input| Optional. When the value is True, the tracker automatically increments. The default value is False.|

## Constraints

- This API must be used together with vf.store_unalign and called after vf.store_unalign.

- To perform unaligned store post-processing based on the valid element count produced by vf.squeeze, use the vf.squeeze_store_unalign_post API.

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
    ureg = vf.load_unalign_init()
    vf.load_unalign_pre(ureg, src_tile)
    reg = vf.load_unalign(ureg, src_tile, 64)
    align_reg = vf.unalign_reg_for_store()
    vf.store_unalign(dst_tile, reg, align_reg, 64, post_update=True)
    vf.store_unalign_post(dst_tile, align_reg, 0, post_update=True)

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
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
