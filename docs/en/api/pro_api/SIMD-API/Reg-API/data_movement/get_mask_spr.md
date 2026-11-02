# vf.get_mask_spr

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:46:14.355Z pushedAt=2026-10-28T11:57:28.975Z nl=2 -->

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

Reads the mask value from the mask registers {MASK1, MASK0} set by pypto_pro.language.set_vec_mask, converts it to the format corresponding to the data type, and writes the result to the return value [mask_reg](../basic_data_structures/mask_reg.md).

The conversion is performed as follows:

- **32-bit width (DT_INT32, DT_UINT32, DT_FP32)**: Reads the 64-bit MASK0 data, replicates each bit to 4 bits, and writes the result to mask_reg.
- **16-bit width (DT_INT16, DT_UINT16, DT_FP16, DT_BF16)**: Reads the complete 128-bit {MASK1, MASK0} data, replicates each bit to 2 bits, and writes the result to mask_reg.

## Prototype

```python
get_mask_spr(width: MaskWidth = MaskWidth.B32) -> preg
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| width | Input | Optional. Mask width, which determines the SPR bit width to read and the expansion method, corresponding to the [MaskWidth](../basic_data_structures/MaskWidth.md) type.<br>- pypto_pro.language.MaskWidth.B32 (default): Reads 64-bit MASK0, expanding each bit to 4 bits.<br>- pypto_pro.language.MaskWidth.B16: Reads 128-bit {MASK1, MASK0}, expanding each bit to 2 bits. |

## Constraints

- Data type constraints:

  | width | Return Value |
  |---|---|
  | pypto_pro.language.MaskWidth.B32 | mask_reg (corresponding to 32-bit width: DT_INT32, DT_UINT32, DT_FP32) |
  | pypto_pro.language.MaskWidth.B16 | mask_reg (corresponding to 16-bit width: DT_INT16, DT_UINT16, DT_FP16, DT_BF16) |

- This API is a compatibility API. It is recommended to use vf.create_mask and vf.update_mask for mask_reg computation.

- Before using this API, select a mode consistent with the mask semantics, and set SPR {MASK1, MASK0} through pypto_pro.language.set_vec_mask or a VF instruction that generates a mask. Use norm mode for bitwise masks and count mode for element counting.

## Return Value

Returns mask_reg, which is read from SPR {MASK1, MASK0} and converted.

## Examples

First set SPR {MASK1, MASK0} using pypto_pro.language.set_vec_mask, then read it into mask_reg using vf.get_mask_spr, and finally use this mask_reg to control the computation:

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    spr_mask = vf.get_mask_spr(width=pl.MaskWidth.B32)
    reg_dst = vf.abs(reg, spr_mask)
    vf.store_align(dst_tile, reg_dst, preg)

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
        pl.set_mask_norm()
        pl.set_vec_mask(0, 0xFFFFFFFF)
        example_vf(in_a, t_out)
        pl.reset_mask()
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
    expected = torch.zeros_like(a)
    expected[:, :32] = torch.abs(a[:, :32])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
