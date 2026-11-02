# vf.pair_reduce_sum

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:06:42.317Z pushedAt=2026-10-28T11:57:29.068Z nl=2 -->

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

Adds each pair of adjacent values in the input src and stores the results in the low-order positions of dst.

$$dstReg_i = srcReg_{2i} + srcReg_{2i+1}$$

## Prototype

```python
pair_reduce_sum(src, preg, mode: Optional[MergeMode] = None) -> dst
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src | Input | Source operand, [reg_tensor](../basic_data_structures/reg_tensor.md). The source operand src and the destination operand dst must have the same data type. Supported data types: DT_FP16 and DT_FP32. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). Elements not filtered by the mask are set to zero in dst. |
| mode | Input | Optional, corresponding to the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not filtered by preg are set to 0 in dst.<br>- pypto_pro.language.MergeMode.MERGING: Not supported. |

## Constraints

None

## Return Value

Returns the destination operand dst, [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are the same as those described for src.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src_reg = vf.load_align(src_tile, 0)
    dst_reg = vf.pair_reduce_sum(src_reg, preg)
    vf.store_align(dst_tile, dst_reg, preg)

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
    expected = torch.zeros([1, 64], device=device, dtype=torch.float32)
    expected[0, :32] = a[0, 0::2] + a[0, 1::2]
    torch.testing.assert_close(out[0, :32], expected[0, :32], rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
