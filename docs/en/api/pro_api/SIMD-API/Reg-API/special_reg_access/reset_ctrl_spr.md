# pypto_pro.language.reset_ctrl_spr

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:12:48.303Z pushedAt=2026-10-28T11:57:29.101Z nl=2 -->

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

Restores the specified bit range in the CTRL special register to the hardware default value. The default value of the CTRL register is 0x1000000000000008.

This API is typically used to restore the default state after the CTRL register is modified through pypto_pro.language.set_ctrl_spr or pypto_pro.language.set_saturation_flag.

## Prototype

```python
reset_ctrl_spr(start_bit: int, end_bit: int, /) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| start_bit | Input | Start bit (0-63), a compile-time constant. |
| end_bit | Input | End bit (0-63), a compile-time constant. |

## Constraints

- The resettable CTRL bits are the same as those of set_ctrl_spr: 6-10, 45, 48, 50, 53, 59, and 60.
- After restoration, the specified bit range in the CTRL register returns to the default value, while other bits remain unaffected.

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
    reg = vf.load_align(src_tile, 0)
    reg_i16 = vf.astype(reg, preg, dtype=pl.DT_INT16, layout=pl.CastLayout.ZERO)
    reg_f32 = vf.astype(reg_i16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])
    pl.reset_ctrl_spr(59, 59)

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32) * 50000
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, 1](a, out)
    torch.npu.synchronize()
    expected = a.clamp(-32768, 32767).to(torch.int16).to(torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
