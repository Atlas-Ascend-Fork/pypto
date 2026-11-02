# pypto_pro.language.get_block_idx

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:33:46.278Z pushedAt=2026-10-28T11:57:29.197Z nl=2 -->

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

Obtains the global index of the current actually executing core within its execution domain, for multi-core control and data offset computation.

## Prototype

```python
pypto_pro.language.get_block_idx() -> int
```

## Parameters

None

## Constraints

None

## Return Value

Returns the current logical core index of the DT_INT64 type, which can be used for integer computation and data indexing within the Kernel. The value range depends on the Kernel execution domain:

- When only the Cube (AIC) or only the Vector (AIV) is started, the range is [0, get_block_num()).
- When both AIC and AIV are started, the AIC-side range is [0, get_block_num()); the AIV-side range is [0, get_subblock_num() × get_block_num()). Under the current 1:2 configuration, the AIV-side range is [0, 2 × get_block_num()).

In the Vector section, this API returns the global AIV logical index, numbered as: physical core number × get_subblock_num() + sub-core number, which can be directly used for data sharding and offset computation. When a hybrid Kernel performs strided partitioning in the Vector section, the total number of work units is get_block_num() × get_subblock_num().

## Examples

### Multi-Core Data Sharding (Pure Vector Kernel)

Two AIVs are actually launched using the Kernel[None, NUM_CORES] form. Each AIV uses get_block_idx() to obtain the global logical index and processes 64 rows of element-wise addition, and uses [pypto_pro.language.printf](../../Utils-API/debugging/printf.md) to print the obtained index value:

```python
import os
import pypto_pro.language as pl
import torch

NUM_CORES = 2


@pl.jit(auto_mutex=True)
def multicore_add_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        vidx = pl.get_block_idx()              # Global logical index of the current AIV.
        num_blocks = pl.get_block_num()        # Number of blocks actually launched.
        pl.printf("block_idx = %d, block_num = %d\n", vidx, num_blocks)
        for tile_idx in pl.range(vidx, 2, num_blocks):
            offset = tile_idx * 64
            cur_a = tile_a.current()
            cur_b = tile_b.current()
            cur_c = tile_c.current()
            pl.load(cur_a, x, [offset, 0])
            pl.load(cur_b, y, [offset, 0])
            pl.add(cur_c, cur_a, cur_b)
            pl.store(z, cur_c, [offset, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.rand([128, 128], device=device, dtype=torch.float16)
    y = torch.rand([128, 128], device=device, dtype=torch.float16)
    z = torch.zeros([128, 128], device=device, dtype=torch.float16)

    multicore_add_kernel[None, NUM_CORES](x, y, z)
    torch.npu.synchronize()

    torch.testing.assert_close(z, x + y, rtol=1e-2, atol=1e-2)
    print(f"max diff = {(z - (x + y)).abs().max().item()}")
```

Output (`=> Vec` is followed by the core number, and the output order across cores is not fixed):

```text
=> Vec 0
block_idx = 0, block_num = 2

=> Vec 1
block_idx = 1, block_num = 2
```

### Return Values of AIC and AIV in a Hybrid Kernel

In a hybrid kernel (AIC:AIV=1:2), the Cube segment returns the physical core ID in the range [0, get_block_num()), and the Vector segment returns the global AIV logical ID in the range [0, get_block_num() × get_subblock_num()). The ID is calculated as: physical core ID × get_subblock_num() + sub-core ID:

```python
import os
import pypto_pro.language as pl
import torch

@pl.jit()
def block_idx_mix_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_cube():
        aic_idx = pl.get_block_idx()
        pl.printf("[cube] block_idx = %d\n", aic_idx)
        pl.setval(out, 0, 1)
    with pl.section_vector():
        aiv_idx = pl.get_block_idx()
        pl.printf("[vector] block_idx = %d\n", aiv_idx)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    out = torch.zeros(1, device=device, dtype=torch.int32)

    block_idx_mix_kernel[None, 2](out)
    torch.npu.synchronize()
```

Output (with 2 AICs and 4 AIVs actually launched: the AIC side returns 0/1, and the AIV side returns 0 to 3):

```text
=> Cube 0
[cube] block_idx = 0

=> Vec 0
[vector] block_idx = 0

=> Vec 1
[vector] block_idx = 1

=> Cube 1
[cube] block_idx = 1

=> Vec 2
[vector] block_idx = 2

=> Vec 3
[vector] block_idx = 3
```
