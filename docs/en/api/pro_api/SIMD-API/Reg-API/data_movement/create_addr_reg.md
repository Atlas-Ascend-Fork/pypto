# vf.create_addr_reg

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:44:05.437Z pushedAt=2026-10-28T11:57:28.961Z nl=2 -->

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

vf.create_addr_reg creates an address offset register (AddrReg) that accumulates the address offset layer by layer in multi-dimensional loops. The AddrReg can be used as the address offset parameter of vf.load_align and vf.store_align, replacing a directly passed integer offset.

The offset is determined by the stride corresponding to each loop axis, and 1 to 4 loop axes are supported. In a loop, the AddrReg offset is automatically accumulated according to the stride corresponding to each loop axis.

## Prototype

```python
create_addr_reg(stride0, stride1=None, stride2=None, stride3=None, dtype: Optional[DType] = None) -> a_reg
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| stride0 | Input | Address offset corresponding to the outermost loop axis, in elements. |
| stride1 | Input | Optional. Address offset corresponding to the second loop axis, in elements. |
| stride2 | Input | Optional. Address offset corresponding to the third loop axis, in elements. |
| stride3 | Input | Optional. Address offset corresponding to the fourth loop axis, in elements. |
| dtype | Input | Optional. Data type corresponding to the template parameter (default: pypto_pro.language.DT_FP32). Determines the element width: 8-bit (DT_INT8, DT_UINT8), 16-bit (DT_INT16, DT_UINT16, DT_FP16, DT_BF16), 32-bit (DT_INT32, DT_UINT32, DT_FP32), 64-bit (DT_INT64, DT_UINT64). |

## Constraints

- Data type constraints:

  Supported data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_INT32, DT_UINT32, DT_FP32, DT_INT64, DT_UINT64.

## Return Value

Returns the a_reg destination operand, the AddrReg address offset register.<br>- The maximum number of AddrRegs is 8.<br>- Due to hardware loop limitations, an AddrReg supports at most 4 loop axes.<br>- An AddrReg can be used only with the vf.load_align and vf.store_align data movement instructions.<br>- When data movement is performed with the address offset set through an AddrReg, the address alignment constraints of the corresponding data movement instruction must be satisfied.

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
