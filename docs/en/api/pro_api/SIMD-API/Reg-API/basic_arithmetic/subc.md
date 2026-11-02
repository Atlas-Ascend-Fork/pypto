# vf.subc

<!-- md-trans-meta sourceCommit=564bbd0d467a53020db44477b1b05ee967581970 translatedAt=2026-10-26T13:22:11.624Z pushedAt=2026-10-28T11:57:28.835Z nl=2 -->

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

This API performs element-wise subtraction on the source operands src0 and src1 and the input borrow value borrow_src according to mask, writes the result to the destination operand dst, and writes the borrow result of each element to borrow. The computation formula is as follows:

$$\{borrow_i, dst_i\} = src0_i - src1_i - borrow\_src_i$$

The borrow flag indicates a subtraction borrow. Subtraction is implemented at the hardware level through two's complement addition. If the bitwise addition of the bitwise inverted results of src0 and src1 with the borrow_src input produces no carry out of the most significant bit (i.e., the subtraction is insufficient), then the corresponding position in borrow (the mask_reg storing the borrow) is set to 1 every 4 bits; otherwise, 0 is written.

## Prototype

```python
subc(src0, src1, borrow_src, preg) -> (borrow, dst)
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src0 | Input | Source operand 0, [reg_tensor](../basic_data_structures/reg_tensor.md). The source operands src0 and src1 and the destination operand dst must have the same data type. Supported data types: DT_INT32 and DT_UINT32. |
| src1 | Input | Source operand 1, [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are the same as those described for src0. |
| borrow_src | Input | Input borrow value, [mask_reg](../basic_data_structures/mask_reg.md). It must be created before use: create it through [vf.create_mask](../mask_operations/create_mask.md), or provide it from the borrow output of a previous vf.subc. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). |

## Constraints

None

## Return Value

Returns a two-tuple (borrow, dst).

- borrow: Output borrow value, [mask_reg](../basic_data_structures/mask_reg.md). It must be created before use: create it through [vf.create_mask](../mask_operations/create_mask.md) as the tuple unpacking target, and it can be used as the borrow_src input of the next vf.subc.
- dst: Destination operand, [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are the same as those described for src0.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    borrow_src = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    borrow = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_a = vf.load_align(src_a, 0, dtype=pl.DT_UINT32)
    reg_b = vf.load_align(src_b, 0, dtype=pl.DT_UINT32)
    borrow, reg_out = vf.subc(reg_a, reg_b, borrow_src, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        example_vf(in_a, in_b, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(0, 50, [1, 64], device=device, dtype=torch.int32)
    b = torch.randint(0, 50, [1, 64], device=device, dtype=torch.int32)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    expected = (a.to(torch.int64) - b.to(torch.int64)).to(torch.int32)
    assert torch.equal(out, expected)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
