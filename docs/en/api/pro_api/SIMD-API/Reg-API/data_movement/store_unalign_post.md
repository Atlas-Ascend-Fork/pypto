# vf.store_unalign_post

<!-- md-trans-meta sourceCommit=9171eae97254d2eec10aded29872099f6cc9ac2d translatedAt=2026-10-26T13:54:47.202Z pushedAt=2026-10-28T11:57:29.011Z nl=2 -->

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

Unaligned store post-processing that processes the remaining unaligned bytes in the unaligned register (UnalignRegForLoad). It must be called after vf.store_unalign.

## Prototype

```python
store_unalign_post(tile, align_reg, stride, post_update: bool = False)
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| tile | Output | Destination operand, a Tile address. The data types of the destination operand and the source operand must be consistent. Supported data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_INT64, DT_UINT64, DT_FP8E4M3FN, DT_FP8E5M2, DT_FP8E8M0, DT_HF8, DT_FP4E2M1, DT_FP4E1M2. |
| align_reg | Input | Alignment tracker register (created by vf.unalign_reg_for_store()). |
| stride | Input | Number of elements to store or an address register. When it is an integer scalar, a vstas instruction is emitted (strided mode), and when post_update = True, it also serves as the address update stride, which is valid only when post_update = True. When it is an AddrReg (created by vf.create_addr_reg), a vsta instruction is emitted (AddrReg mode), which must be paired with the AddrReg mode (vstu) of vf.store_unalign. |
| post_update | Input | Optional. When True, the tracker advances automatically. Defaults to False. |

## Constraints

- Must be paired with vf.store_unalign and called after vf.store_unalign.

- To perform unaligned store post-processing based on the number of valid elements of vf.squeeze, use the vf.squeeze_store_unalign_post API.

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
