# vf.create_addr_reg

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

vf.create_addr_reg creates an address offset register (AddrReg) that accumulates address offsets layer by layer in multi-dimensional loops. AddrReg can be used as the address offset argument for vf.load_align and vf.store_align, replacing a directly passed integer offset.

The offset is determined by the stride corresponding to each loop axis, and supports 1 to 4 loop axes. Within the loop, the AddrReg offset is automatically accumulated according to the stride of each loop axis.

## Prototype

```python
create_addr_reg(stride0, stride1=None, stride2=None, stride3=None, /, *, dtype: Optional[DType] = None) -> a_reg
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| stride0 | Input| Address offset corresponding to the outermost loop axis, in units of elements.|
| stride1 | Input| (Optional) Address offset corresponding to the second loop axis, in units of elements.|
| stride2 | Input| (Optional) Address offset corresponding to the third loop axis, in units of elements.|
| stride3 | Input| (Optional) Address offset corresponding to the fourth loop axis, in units of elements.|
| dtype | Input| (Optional) Data type corresponding to the template parameter (pypto_pro.language.DT_FP32 by default). This parameter determines the element width: 8-bit width (DT_INT8 and DT_UINT8), 16-bit width (DT_INT16, DT_UINT16, DT_FP16, and DT_BF16), 32-bit width (DT_INT32, DT_UINT32, and DT_FP32), or 64-bit width (DT_INT64 and DT_UINT64).|

## Constraints

- Data type constraints:

  The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_INT64, and DT_UINT64.

## Return Value

Returns the destination operand a_reg and the address offset register AddrReg.<br>- The maximum number of AddrReg registers is 8.<br>- Due to hardware loop restrictions, AddrReg supports a maximum of four loop axes.<br>- AddrReg can be used only with the vf.load_align and vf.store_align data movement instructions.<br>- When AddrReg is used to set address offsets for data movement, the address alignment constraints of the corresponding data movement instruction must be satisfied.

## Examples

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one_repeat_size = 64
    repeat_times = 2
    for i in pl.range(0, repeat_times, 1):
        a_reg = vf.create_addr_reg(one_repeat_size, dtype=pl.DT_FP32)
        reg = vf.load_align(src_tile, a_reg)
        vf.store_align(dst_tile, reg, preg, a_reg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[1])
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
    a = torch.randn([1, 128], device=device, dtype=torch.float32)
    out = torch.empty([1, 128], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
