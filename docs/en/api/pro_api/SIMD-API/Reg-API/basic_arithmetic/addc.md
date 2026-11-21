# vf.addc

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

Based on the mask, this API performs element-wise addition on the source operands src0 and src1 and the input carry carry_src, writes the addition result to the destination operand dst, and writes the carry result for each element to carry. The formula is as follows:

$$\{carry_i, dst_i\} = src0_i + src1_i + carry\_src_i$$

Carry flag indicates the carry from addition. If a carry occurs from the most significant bit after bitwise addition of the inputs src0, src1, and carry_src, a value of 1 is set for every 4 bits at the corresponding position in carry (a mask_reg that stores the carry); otherwise, 0 is written.

Taking the computation of –1 + 8 = 7 with int64_t data as an example, the applicable scenario of the vf.addc API is shown below:

**Figure 1** Applicable scenario of vf.addc

![Applicable scenario of vf.addc](../../../figures/addc_scenario.svg)

## Prototype

```python
addc(src0, src1, carry_src, preg, /) -> (carry, dst)
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| src0 | Input| Source operand 0, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). The source operands src0 and src1 and the destination operand dst must have the same data type. The supported data types are DT_INT32 and DT_UINT32.|
| src1 | Input| Source operand 1, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are the same as those for src0.|
| carry_src | Input| Input carry value, which is of type [mask_reg](../basic_data_structures/mask_reg.md). It must be created before use by calling [vf.create_mask](../mask_operations/create_mask.md) or provided by the carry output of the previous vf.addc operation.|
| preg | Input| [mask_reg](../basic_data_structures/mask_reg.md).|

## Constraints

None

## Return Value

Returns a 2-tuple (carry, dst).

- carry: output carry value, which is of type [mask_reg](../basic_data_structures/mask_reg.md). It must be created before use by calling [vf.create_mask](../mask_operations/create_mask.md). After it is created, it is used as a tuple unpacking target and can serve as the carry_src input for the next vf.addc call.
- dst: destination operand, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are the same as those of src0.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    carry_src = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    carry = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_a = vf.load_align(src_a, 0, dtype=pl.DT_UINT32)
    reg_b = vf.load_align(src_b, 0, dtype=pl.DT_UINT32)
    carry, reg_out = vf.addc(reg_a, reg_b, carry_src, preg)
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
    expected = (a.to(torch.int64) + b.to(torch.int64) + 1).to(torch.int32)
    assert torch.equal(out, expected)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
