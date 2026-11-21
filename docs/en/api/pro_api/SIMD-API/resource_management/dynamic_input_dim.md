# pypto_pro.language.dynamic_input_dim

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

When there are multiple input tensors with the same dimension but varying quantities and shapes, you can use the packing method described in [Constraints](#constraints) to combine them into a dynamic input based on certain requirements. The start addresses of these input tensors are recorded using pointers. This API reads the size of a specified dimension of a specified Tensor from the dynamic input.

This API is often used together with [pypto_pro.language.dynamic_input_ptr](dynamic_input_ptr.md). After obtaining the size of a specified dimension of a specified Tensor, you can access the Tensor data through the address.

## Prototype

```python
pypto_pro.language.dynamic_input_dim(x: Ptr, i: int, axis: int, /) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description                                                                                                             |
| ------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| x      | Input     | Input pointer. The type is dtype in [pypto_pro.language.Ptr](../basic_data_structures/Ptr.md). For details about how to pack multiple Tensors, see the constraints.|
| i      | Input     | Tensor index, which is an integer constant or a runtime integer scalar. The value range is [0, N), where N indicates the number of Tensors passed in this call.                                |
| axis   | Input     | Dimension index, which is an integer constant or a runtime integer scalar. The value range is [0, dim), where dim indicates the dimension of the target Tensor.                                    |

## Constraints

- The following figure shows how to pack multiple Tensors that contain shape information. For details about the code implementation, see [Examples](#examples).

  ```text
           ┌──────────────┐
      w[0] │    offset    │  Byte offset of the address array = (1 + 2N) * 8
           ├──────────────┤
      w[1] │ dim | count  │  Metadata word of structure 0: lower 32 bits for dim(=1) and upper 32 bits for N
      w[2] │    shape     │  Length of axis 0 of Tensor 0
      w[3] │      0       │  Metadata word of structure 1 (set to 0)
      w[4] │    shape     │  Length of axis 0 of Tensor 1
       ... │     ...      │  ...
           ├──────────────┤
   w[1+2N] │    addr_0    │  Start address of Tensor 0
   w[2+2N] │    addr_1    │  Start address of Tensor 1
       ... │     ...      │  ...
     w[3N] │  addr_{N-1}  │  Start address of Tensor N-1
           └──────────────┘

  For example, if N=2, dim=1, and shape=11, the array is [40, (2 << 32) | 1, 11, 0, 11, addr_0, addr_1].
  ```

## Return Value

Returns the length of the axis-th dimension of the i-th tensor, as an INT64 scalar.

## Examples

```python
import os

import pypto_pro.language as pl
import torch

MAX_AXIS = 4


def _build_list_tensor_desc_dim1(addrs, shape0):
    """Dynamic input with dim=1: [dataPtrOffset, N shape structures of (1+dim)=2 words, addr_0..addr_{N-1}]

    Address retrieval still reads a contiguous address array. The shape block is located between the start of this parameter and the address array. Therefore, the offset of the address array varies with N.
    """
    n = len(addrs)
    struct_sz = 2  # 1 (metadata word) + dim(=1)
    data_ptr_offset = 8 + n * struct_sz * 8
    w = [0] * (1 + n * struct_sz + n)
    w[0] = data_ptr_offset
    w[1] = (1 & 0xFFFFFFFF) | (n << 32)  # Metadata word of structure 0: lower 32 bits for dim=1 and upper 32 bits for count
    w[2] = shape0  # shape[0] of structure 0
    for i in range(1, n):
        w[1 + i * struct_sz] = 0  # Metadata words of other structures, set to 0
        w[1 + i * struct_sz + 1] = shape0
    addr_off = data_ptr_offset // 8
    for i, a in enumerate(addrs):
        w[addr_off + i] = a
    return torch.tensor(w, dtype=torch.int64)


@pl.jit(auto_mutex=True)
def dynamic_input_dim_read(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_INT64],
    dim: pl.DT_INT32,
):
    with pl.section_vector():
        out_t = pl.make_tensor(out, [1, 4], [4, 1], dtype=pl.DT_INT64)
        # The axis value is in the range [0, dim). If it is out of bounds, execution will be terminated on the device.
        for axis in pl.range(0, dim):
            pl.setval(out_t, axis, pl.dynamic_input_dim(desc, 1, axis))


def run():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dim, shape0 = 1, 11
    # Two independently allocated tensors. Each tensor in this parameter records only one axis length.
    tensors = [torch.zeros(16, device=device, dtype=torch.float32) for _ in range(2)]
    desc = _build_list_tensor_desc_dim1([t.data_ptr() for t in tensors], shape0).to(device)
    out = torch.full((1, MAX_AXIS), -1, device=device, dtype=torch.int64)
    dynamic_input_dim_read[None, 1](desc, out, dim)
    torch.npu.synchronize()
    got = out.cpu().flatten().tolist()[:dim]
    torch.testing.assert_close(got, [shape0], rtol=0, atol=0)
    print(f"dim={dim} shape0={shape0} PASSED")


if __name__ == "__main__":
    run()
```
