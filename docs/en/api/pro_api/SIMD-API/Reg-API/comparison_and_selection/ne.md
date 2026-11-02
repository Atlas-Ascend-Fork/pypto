# vf.ne

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:37:52.245Z pushedAt=2026-10-28T11:57:28.920Z nl=2 -->

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

Compares the two operands src0 and src1 element by element to determine whether they are not equal, and writes the comparison result to the corresponding bits of the destination operand dst_mask. If the comparison result is true, the corresponding bit of the output result is 1; otherwise, it is 0.

The second parameter can be a scalar or a reg_tensor. The API automatically identifies the type and dispatches it to the corresponding hardware instruction.

$$dstReg_i = \begin{cases} 1 & \text{if } src0_i \neq src1_i \\ 0 & \text{otherwise} \end{cases}$$

## Prototype

```python
ne(src0, src1, preg) -> dst_mask
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src0 | Input | Source operand, [reg_tensor](../basic_data_structures/reg_tensor.md). Supported data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32. src0 and src1 can be the same reg_tensor. |
| src1 | Input | Comparison operand, which can be a scalar or a [reg_tensor](../basic_data_structures/reg_tensor.md) with the same data type as src0. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md), which specifies the range of elements involved in the comparison. Elements not selected through the preg parameter are set to zero in the destination operand. |

## Constraints

None

## Return Value

Returns the destination [mask_reg](../basic_data_structures/mask_reg.md) dst_mask, which stores the comparison result.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    dst_mask = vf.ne(reg_a, reg_b, preg)
    reg_out = vf.select(reg_a, reg_b, dst_mask)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
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
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.where(a != b, a, b), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
