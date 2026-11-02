# pypto_pro.language.get_block_num

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:36:22.787Z pushedAt=2026-10-28T11:57:29.206Z nl=2 -->

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

Obtains the number of logical Blocks actually launched this time, for multi-core control and data offset computation.

## Prototype

```python
pypto_pro.language.get_block_num() -> int
```

## Parameters

None

## Constraints

None

## Return Value

Returns a scalar value of the DT_INT64 type, indicating the number of Blocks actually launched after core limiting. It can be used for integer computation and data indexing within the Kernel.
For each JIT launch, the C++ launcher queries the effective resource limits of the actual Stream and limits the `block_dim` requested by the Host according to the Kernel execution domain and the pairing ratio.
Therefore, the return value may be smaller than the value requested by the Host. Data partitioning should use the return value of this API as the loop step to avoid missing tasks.

When only Cube (AIC) or only Vector (AIV) is launched, this value equals the number of logical cores in the execution domain.
In a mixed Kernel with an AIC:AIV ratio of 1:2, this value indicates the number of logical Blocks. The number of AIC logical cores is `get_block_num()`,
and the number of AIV logical cores is `get_block_num() * get_subblock_num()`.

## Examples

### Splitting by the Actual Number of Blocks

Use Kernel[None, NUM_CORES] to request up to two logical blocks, with each AIV processing 64 rows of tiles in strides equal to the value returned by get_block_num(). Use [pypto_pro.language.printf](../../Utils-API/debugging/printf.md) to print the actual number of blocks started:

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
        num_blocks = pl.get_block_num()        # Actual number of blocks started.
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

Output (`=> Vec` is followed by the core number, and the output order across multiple cores is not fixed):

```text
=> Vec 0
block_idx = 0, block_num = 2

=> Vec 1
block_idx = 1, block_num = 2
```

### Return Value Under Limited-Core Launch

The number of launched blocks is the requested upper bound, and the actual number launched may be smaller. Reusing the Kernel and x/y in `__main__` from the previous example, change only the number of launched blocks to 1:

```python
    z = torch.zeros([128, 128], device=device, dtype=torch.float16)
    multicore_add_kernel[None, 1](x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y, rtol=1e-2, atol=1e-2)  # Still covers all 128 rows.
```

Output:

```text
=> Vec 0
block_idx = 0, block_num = 1
```
