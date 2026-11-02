# pypto_pro.language.get_ctrl_spr

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:10:48.456Z pushedAt=2026-10-28T11:57:29.091Z nl=2 -->

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

Reads the value of a specified bit range in the CTRL special register and returns the current value of that bit range.

## Prototype

```python
get_ctrl_spr(start_bit: int, end_bit: int, /) -> int
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| start_bit | Input | Start bit (0-63) of the special register to read, a compile-time constant. |
| end_bit | Input | End bit (0-63) of the special register to read, a compile-time constant. |

## Constraints

- The read range is not limited by the writable bits; any bit range can be read.

## Return Value

Returns an int value, which is the value of the [start_bit, end_bit] bit range in the CTRL register.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    global_mode = pl.get_ctrl_spr(60, 60)
    float_sat = pl.get_ctrl_spr(48, 48)
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.store(out, in_a, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, 1](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
