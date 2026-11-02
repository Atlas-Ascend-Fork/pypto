# pypto_pro.language.get_subblock_idx

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:34:51.203Z pushedAt=2026-10-28T11:57:29.200Z nl=2 -->

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

Obtains the subblock index of the AIC or AIV in the current logical AI Core.

## Prototype

```python
pypto_pro.language.get_subblock_idx() -> int
```

## Parameters

None

## Constraints

None

## Return Value

Returns the subblock index of the DT_INT64 type, which can be used for integer computation and data indexing within the kernel. The value range is [0, get_subblock_num()). In a mixed kernel with an AIC-to-AIV ratio of 1:2, the two AIVs corresponding to the same logical block return 0 and 1, respectively.

## Examples

### Reading the Subblock Index in a Pure Vector Kernel

In a pure Vector Kernel, each Block is an AIV (get_subblock_num() returns 1), and get_subblock_idx() is always 0:

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit(auto_mutex=True)
def subblock_add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_sum = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        sub_idx = pl.get_subblock_idx()
        pl.printf("subblock_idx = %d, subblock_num = %d\n", sub_idx, pl.get_subblock_num())
        cur_x = tile_x.current()
        cur_y = tile_y.current()
        cur_sum = tile_sum.current()
        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.add(cur_sum, cur_x, cur_y)
        pl.store(out, cur_sum, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.randn([64, 64], device=device, dtype=torch.float32)
    y = torch.randn([64, 64], device=device, dtype=torch.float32)
    out = torch.zeros([64, 64], device=device, dtype=torch.float32)

    subblock_add_kernel(x, y, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + y, rtol=1e-5, atol=1e-5)
    print(f"max diff = {(out - (x + y)).abs().max().item()}")
```

Output:

```text
=> Vec 0
subblock_idx = 0, subblock_num = 1
```

### Sub-core Number and Conditional Execution in a Mixed Kernel

In the Vector section of a mixed kernel, get_block_idx() returns the global AIV logical index for data partitioning, and get_subblock_idx() distinguishes the two AIVs within the same logical block. You can perform conditional execution based on the sub-core number, for example, to let only sub-core 0 write the result:

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit()
def subblock_cond_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_cube():
        pass
    with pl.section_vector():
        block_idx = pl.get_block_idx()      # Global AIV logical index: 0 to 3.
        sub_idx = pl.get_subblock_idx()     # AIV number within the block: 0 or 1.
        pl.printf("block_idx = %d, subblock_idx = %d\n", block_idx, sub_idx)
        # Conditional execution: let only sub-core 0 of each logical block write the result.
        if sub_idx == 0:
            pl.printf("sub-core 0 of core %d writes output\n", block_idx // pl.get_subblock_num())
            pl.setval(out, 0, 1)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    out = torch.zeros(1, device=device, dtype=torch.int32)

    subblock_cond_kernel[None, 2](out)
    torch.npu.synchronize()
```

Output (two logical blocks are launched; the `=> Cube` entries are from the Cube section, which prints nothing in this example):

```text
=> Cube 0

=> Vec 0
block_idx = 0, subblock_idx = 0
sub-core 0 of core 0 writes output

=> Vec 1
block_idx = 1, subblock_idx = 1

=> Cube 1

=> Vec 2
block_idx = 2, subblock_idx = 0
sub-core 0 of core 1 writes output

=> Vec 3
block_idx = 3, subblock_idx = 1
```
