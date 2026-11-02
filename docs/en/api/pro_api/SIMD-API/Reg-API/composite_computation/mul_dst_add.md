# vf.mul_dst_add

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:41:42.743Z pushedAt=2026-10-28T11:57:28.947Z nl=2 -->

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

This API performs a fused multiply-add (FMA) operation element-wise on dst, src0, and src1 according to preg. It multiplies dst by src0, adds src1 to the product, and writes the result back to dst. The formula is as follows:

$$
dst_i = dst_i \times src0_i + src1_i
$$

Because the multiplication and addition are completed in a single instruction, the intermediate product is not truncated or rounded due to register width limits. Therefore, the precision is higher than the step-by-step approach of calling vf.mul first and then vf.add.

The dst register is both read (as the multiplicand) and written (to store the result), so it must be pre-initialized before the call.

## Prototype

```python
mul_dst_add(src0, src1, preg, mode: Optional[MergeMode] = None) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src0 | Input | Source operand 0 (multiplier), [reg_tensor](../basic_data_structures/reg_tensor.md). The data types of source operands src0 and src1 and the destination operand dst must be consistent. Supported data types: DT_FP16, DT_BF16, DT_FP32. |
| src1 | Input | Source operand 1 (addend), [reg_tensor](../basic_data_structures/reg_tensor.md), with the same data type as src0. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). |
| mode | Input | Optional, corresponding to the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not selected by preg are set to 0 in dst.<br>- pypto_pro.language.MergeMode.MERGING is currently not supported. |

## Constraints

None

## Return Value

Returns the dst destination/accumulator operand, [reg_tensor](../basic_data_structures/reg_tensor.md), with the supported data types as described for src0. Before the computation, it participates as the multiplicand; after the computation, it stores the result. It must be pre-initialized before the call.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src0_tile, src1_tile, dst_tile, out_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src0_tile, 0)
    reg_b = vf.load_align(src1_tile, 0)
    reg_out = vf.load_align(dst_tile, 0)
    reg_out = vf.mul_dst_add(reg_a, reg_b, preg)
    vf.store_align(out_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    c: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    in_c_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    in_c = in_c_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x300, mutex_ids=[3])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.load(in_c, c, [0, 0])
        example_vf(in_a, in_b, in_c, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    c = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, c, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, c * a + b, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
