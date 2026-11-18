# pypto_pro.language.make_list_tensor_desc

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

实现了输入多个Tensor的打包封装，打包后作为pl.Ptr[dtype]参数传入，配合[pypto_pro.language.dynamic_input_ptr](dynamic_input_ptr.md)和[pypto_pro.language.dynamic_input_dim](dynamic_input_dim.md)解码使用。

## 函数原型

```python
pypto_pro.language.make_list_tensor_desc(
    addrs: Sequence[int],
    shape: Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> torch.Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                                                    |
| ------ | --------- | ------------------------------------------------------------------------------------------------------- |
| addrs  | 输入      | 各个Tensor的起始地址，Sequence[int]类型，至少包含1个地址。                                              |
| shape  | 输入      | 每个Tensor携带的形状，可选。`None`（默认）构造不含形状信息；扁平Sequence[int]为所有Tensor共享的同一形状；嵌套Sequence[Sequence[int]]逐Tensor指定形状，长度须与addrs一致，且各Tensor维数相同。各轴长度须为正整数（不接受0、负数、布尔值）；shape不能为空列表，dim=0形式须省略shape参数而非传入`[]`，否则校验报错。 |

## 约束说明

- `shape=None`时构造不含形状信息的dim=0形式，多个Tensor不包含形状信息的打包方式示意图如下，代码实现可见[调用示例](#调用示例)：

  ```text
           ┌──────────────┐
      w[0] │      24      │  地址数组的字节偏移（固定24）
           ├──────────────┤
      w[1] │ dim | count  │  低32 bit为dim(=0)，高32 bit为Tensor个数N
      w[2] │  0xFFFFFFFF  │  保留字
           ├──────────────┤
      w[3] │    addr_0    │  第0个Tensor的起始地址
      w[4] │    addr_1    │  第1个Tensor的起始地址
       ... │     ...      │  ...
    w[2+N] │  addr_{N-1}  │  第N-1个Tensor的起始地址
           └──────────────┘

  以N=2、dim=0为例，该数组为[24, (2 << 32) | 0, 0xFFFFFFFF, addr_0, addr_1]。
  ```
- `shape`为非空列表时构造含形状信息的dim形式：扁平列表（如`[3, 5]`）时dim=`len(shape)`，所有Tensor共享该形状；嵌套列表（如`[[33, 34, 35], [23, 24, 25]]`）时逐Tensor携带各自的轴长，但dim为全局唯一、struct步长统一，因此各Tensor的维数必须相同。多个Tensor包含形状信息的打包方式示意图如下：

  ```text
           ┌──────────────┐
      w[0] │    offset    │  地址数组的字节偏移 = (1 + 2N) * 8
           ├──────────────┤
      w[1] │ dim | count  │  第0个结构的元信息字：低32 bit为dim(=1)，高32 bit为N
      w[2] │    shape     │  第0个Tensor的第0轴长度
      w[3] │      0       │  第1个结构的元信息字（填0）
      w[4] │    shape     │  第1个Tensor的第0轴长度
       ... │     ...      │  ...
           ├──────────────┤
   w[1+2N] │    addr_0    │  第0个Tensor的起始地址
   w[2+2N] │    addr_1    │  第1个Tensor的起始地址
       ... │     ...      │  ...
     w[3N] │  addr_{N-1}  │  第N-1个Tensor的起始地址
           └──────────────┘

  以N=2、dim=1、shape=11为例，该数组为[40, (2 << 32) | 1, 11, 0, 11, addr_0, addr_1]。
  ```

## 返回值说明

返回int64类型的`torch.Tensor`，内容为动态输入的打包信息，启动kernel前须`.to(device)`迁移至设备。

## 调用示例

```python
import os

import pypto_pro.language as pl
import torch

P = 256


@pl.jit(auto_mutex=True)
def dynamic_input_ptr_sum(
    desc: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_FP32],
):
    out_t = pl.make_tensor(out, [1, P], [P, 1], dtype=pl.DT_FP32)
    tt = pl.TileType(shape=[1, P], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    src_db = pl.make_tile_group(type=tt, addrs=[0x1000], mutex_ids=[0])
    acc_db = pl.make_tile_group(type=tt, addrs=[0x2000], mutex_ids=[1])
    with pl.section_vector():
        acc = acc_db.next()
        pl.set_validshape(acc, [1, P])
        # 取第 0 个张量的地址，包装成 Tensor 视图后搬入 UB
        p0 = pl.dynamic_input_ptr(desc, 0)
        pl.load(acc, pl.make_tensor(p0, [1, P], [P, 1], dtype=pl.DT_FP32), [0, 0])
        # 取第 1 个张量的地址，搬入 UB 后累加到第 0 个张量上
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
    # 两个张量独立分配（地址不连续），全部由该参数索引
    tensors = [torch.randn(P, device=device, dtype=torch.float32) for _ in range(2)]
    desc = pl.make_list_tensor_desc([t.data_ptr() for t in tensors]).to(device)
    out = torch.full((1, P), -1.0, device=device, dtype=torch.float32)
    dynamic_input_ptr_sum[None, 1](desc, out)
    torch.npu.synchronize()
    ref = torch.stack(tensors).sum(0, keepdim=True)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    print("PASSED")


if __name__ == "__main__":
    run()
```
