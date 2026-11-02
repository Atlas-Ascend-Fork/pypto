# vf.reduce_min

<!-- md-trans-meta sourceCommit=7d0ab31d84c294a0a8327f597d2bdee012acb22d translatedAt=2026-10-26T14:08:06.482Z pushedAt=2026-10-28T11:57:29.075Z nl=2 -->

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

reg_tensor minimum reduction: Computes the minimum value of all active elements (elements selected by preg) in the source register src, writes the result to the first element dst[0] of the destination register, writes the index of the first minimum value to dst[1], and sets the remaining elements to zero.

The computation process of the minimum reduction and the index storage method are shown in the following figure:

**Figure 1** reduce_min reduction index diagram

![reg_reduce_index diagram](../../../figures/reg_reduce_index.svg)

## Prototype

```python
reduce_min(src, preg, datablock: bool = False, merge_mode: Optional[MergeMode] = None)
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| src | Input | Source operand, [reg_tensor](../basic_data_structures/reg_tensor.md). The source operand src and the destination operand dst must have the same data type. Supported data types: DT_INT16, DT_UINT16, DT_FP16, DT_INT32, DT_UINT32, DT_FP32. |
| preg | Input | [mask_reg](../basic_data_structures/mask_reg.md). When no elements participate in the computation (the mask is empty), the maximum value of the data type is written to dst[0]. |
| datablock | Input | Optional. Determines the working mode of the API. When True, reduction is performed at datablock granularity (corresponding to the vcgmin instruction). Defaults to False. When datablock=True, datablock-granularity reduction is enabled, and each datablock is reduced independently: for 32-bit types (DT_INT32, DT_UINT32, DT_FP32), every 16 elements form a datablock; for 16-bit types (DT_INT16, DT_UINT16, DT_FP16), every 32 elements form a datablock. Each datablock computes its minimum value, and the results are written sequentially to the lowest bits of dst. |
| merge_mode | Input | Optional. Corresponds to the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not selected by preg are set to zero in dst.<br>- pypto_pro.language.MergeMode.MERGING is currently not supported. |

## Constraints

- When datablock=True, the supported data types are: DT_INT16, DT_UINT16, DT_FP16, DT_INT32, DT_UINT32, DT_FP32.

## Return Value

Returns the destination [reg_tensor](../basic_data_structures/reg_tensor.md). The supported data types are the same as those described for src. The reduction result is written to the first element dst[0], and the index is written to dst[1].

- When multiple minimum values exist, the index of the first minimum value is stored in dst[1].

- If the input data contains nan, the nan of that data type is written to dst[0], and the index of the first nan is stored in dst[1].

- min(-0, +0) = -0.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src0 = vf.load_align(src_tile, 0)
    min0 = vf.reduce_min(src0, preg_all)
    vf.store_align(dst_tile, min0, preg_all)

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
    torch.testing.assert_close(out[0, 0], torch.min(a), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
