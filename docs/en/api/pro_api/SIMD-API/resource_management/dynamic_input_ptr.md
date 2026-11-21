# pypto_pro.language.dynamic_input_ptr

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

When there are multiple input tensors with varying quantities and shapes, you can use the packing method described in [Constraints](#constraints) to combine them into a dynamic input based on certain requirements. The start addresses of these input tensors are recorded using pointers. This API reads the start address of a specified Tensor from the dynamic input. The element type of the returned pointer is determined by the declared dtype.

This API is often used to traverse all tensors in sequence. It works with [pypto_pro.language.make_tensor](make_tensor.md) to wrap the address obtained each time into a Tensor view. To read the shape of each tensor, use [pypto_pro.language.dynamic_input_dim](dynamic_input_dim.md).

## Prototype

```python
pypto_pro.language.dynamic_input_ptr(x: Ptr, i: int, /) -> Ptr
```

## Parameters

| Parameter| Input/Output| Description                                                                                                             |
| ------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| x      | Input     | Input pointer. The type is dtype in [pypto_pro.language.Ptr](../basic_data_structures/Ptr.md). For details about how to pack multiple Tensors, see the constraints.|
| i      | Input     | Tensor index, which is an integer constant or a runtime integer scalar. The value range is [0, N), where N indicates the number of Tensors passed in this call.                                |

## Constraints

- The following figure shows how to pack multiple Tensors that do not contain shape information. For details about the code implementation, see [Examples](#examples).

  ```text
           ┌──────────────┐
      w[0] │      24      │  Byte offset of the address array (fixed at 24)
           ├──────────────┤
      w[1] │ dim | count  │  Lower 32 bits for dim (=0) and upper 32 bits for the number of Tensors N
      w[2] │  0xFFFFFFFF  │  Reserved word
           ├──────────────┤
      w[3] │    addr_0    │  Start address of Tensor 0
      w[4] │    addr_1    │  Start address of Tensor 1
       ... │     ...      │  ...
    w[2+N] │  addr_{N-1}  │  Start address of Tensor N-1
           └──────────────┘

  For example, if N=2 and dim=0, the array is [24, (2 << 32) | 0, 0xFFFFFFFF, addr_0, addr_1].
  ```

## Return Value

Returns the address of the i-th tensor. The type is pypto_pro.language.Ptr, and the element type is the same as that of each tensor.

## Examples

```python
import os

import pypto_pro.language as pl
import torch

P = 256


def _build_list_tensor_desc(addrs):
    """Dynamic input with dim=0: [dataPtrOffset, (count<<32)|dim, 0xffffffff, addr_0..addr_{N-1}]"""
    n = len(addrs)
    w = [0] * (3 + n)
    w[0] = 24  # Byte offset from the start to the address array, that is, the address array starts from w[3].
    w[1] = (n << 32) & 0xFFFFFFFFFFFFFFFF  # Lower 32 bits for dim=0 and upper 32 bits for count
    w[2] = 0xFFFFFFFF  # Reserved word
    for i, a in enumerate(addrs):
        w[3 + i] = a
    return torch.tensor(w, dtype=torch.int64)


@pl.jit(auto_mutex=True)
def dynamic_input_ptr_sum(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_FP32],
):
    out_t = pl.make_tensor(out, [1, P], [P, 1], dtype=pl.DT_FP32)
    tt = pl.TileType(shape=[1, P], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    src_db = pl.make_tile_group(type=tt, addrs=[0x0000, 0x1000], mutex_ids=[0, 1])
    acc_db = pl.make_tile_group(type=tt, addrs=[0x2000], mutex_ids=[2])
    with pl.section_vector():
        acc = acc_db.next()
        pl.set_validshape(acc, [1, P])
        # Retrieve the address of tensor 0, wrap it into a Tensor view, and load it into UB.
        p0 = pl.dynamic_input_ptr(desc, 0)
        pl.load(acc, pl.make_tensor(p0, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
        # Retrieve the address of tensor 1, load it into UB, and accumulate it into tensor 0.
        t = src_db.next()
        pl.set_validshape(t, [1, P])
        p1 = pl.dynamic_input_ptr(desc, 1)
        pl.load(t, pl.make_tensor(p1, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
        pl.add(acc, acc, t)
        pl.store(out_t, acc, [0, 0])


def run():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    # The two tensors are allocated independently (with non-contiguous addresses) and are indexed by this parameter.
    tensors = [torch.randn(P, device=device, dtype=torch.float32) for _ in range(2)]
    desc = _build_list_tensor_desc([t.data_ptr() for t in tensors]).to(device)
    out = torch.full((1, P), -1.0, device=device, dtype=torch.float32)
    dynamic_input_ptr_sum[None, 1](desc, out)
    torch.npu.synchronize()
    ref = torch.stack(tensors).sum(0, keepdim=True)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    print("PASSED")


if __name__ == "__main__":
    run()
```
