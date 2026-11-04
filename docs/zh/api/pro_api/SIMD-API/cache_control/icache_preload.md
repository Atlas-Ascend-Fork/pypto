# pypto_pro.language.system.icache_preload

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

将调用位置之后的Kernel指令预加载到指令缓存（Instruction Cache，ICache），用于减少后续指令执行时的ICache Miss。

该接口可在AIC或AIV代码段中使用，预加载范围从接口调用位置之后开始，长度由prefetch_len指定。

## 函数原型

```python
pypto_pro.language.system.icache_preload(prefetch_len: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| prefetch_len | 输入 | 整型常量或运行时整型Scalar，表示预加载长度，单位为2 KiB。在AIC代码段中的取值范围为[0, 15]，在AIV代码段中的取值范围为[0, 7]。 |

## 返回值说明

无。

## 约束说明

无。

## 调用示例

```python
import os

import pypto_pro.language as pl
import torch
import torch_npu  # noqa: F401


@pl.jit(auto_mutex=True)
def icache_preload_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[64, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_x = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[1])

    with pl.section_vector():
        x_tile = tile_x.current()
        out_tile = tile_out.current()
        pl.system.icache_preload(2)
        pl.load(x_tile, x, [0, 0])
        pl.add(out_tile, x_tile, x_tile)
        pl.store(out, out_tile, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    x = torch.randn((64, 64), device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    icache_preload_kernel[None, 1](x, out)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + x, rtol=1e-4, atol=1e-4)
```
