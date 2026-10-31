# pypto_pro.language.dump_data

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

调测打印接口，用于打印GM Tensor或Tile的内容，支持全量打印和窗口打印。

- 输入为Tensor（GM全局内存张量）时，打印GM上的Tensor数据
- 输入为Tile（通过make_tile/make_tile_group分配）时，打印Tile数据

Tile通过[pypto_pro.language.make_tile](../../SIMD-API/resource_management/make_tile.md)或[pypto_pro.language.make_tile_group](../../SIMD-API/resource_management/make_tile_group.md)创建。

打印结果直接输出到终端。全量和窗口均只输出请求范围内的数据，并保留对应layout的物理存储顺序。

## 函数原型

```python
pypto_pro.language.dump_data(
    data: Union[Tensor, Tile],
    offsets: Optional[Sequence[int]] = None,
    shapes: Optional[Sequence[int]] = None,
    *,
    workspace: Optional[Tensor] = None,
    loc: bool = False,
    flag: Optional[str] = None,
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| data | 输入 | 要打印的Tensor或Tile。Tensor支持ND和NZ布局，暂不支持pypto_pro.language.DT_INT4、pypto_pro.language.DT_UINT4、pypto_pro.language.DT_FP4和pypto_pro.language.DT_HF4；Tile不支持L0A Buffer、L0B Buffer、L0A_MX Buffer和L0B_MX Buffer。 |
| offsets | 输入 | 可选，窗口起始偏移（各维）。取值为由整型常量或运行时整型标量表达式组成的序列（列表或元组），长度须等于数据的维数；须与shapes同时提供。NZ Tensor窗口的N方向offset须按dtype对应的C0对齐。 |
| shapes | 输入 | 可选，窗口大小（各维）。取值为由整型常量或运行时整型标量表达式组成的序列（列表或元组），长度须等于数据的维数；须与offsets同时提供。 |
| workspace | 输入 | 保留参数，无需传入。 |
| loc | 输入 | 可选，是否在输出前打印源文件/行号，取值为True或False（默认）。 |
| flag | 输入 | 可选，字符串标签，必须是编译时常量，不能是运行时变量。提供时在dump输出前单独打印一行[flag] 标签内容，用于区分多个dump点的输出。 |

## 约束说明

Runtime默认按每个AIC/AIV核分配32 KB打印FIFO，dump_data和设备侧printf共享该空间。可在首次使用NPU前将每核容量配置为1 KB～64 MB，且须为8字节的整数倍；JIT和二进制Kernel的配置方式相同：

```python
import pypto

pypto.set_debug_options(simd_printf_fifo_size_per_core=256 * 1024)

import pypto_pro.language as pl
import torch
import torch_npu
```

该配置不改变Kernel参数或二进制内容。

## 返回值说明

无。

## 调用示例

### Tensor输入

```python
import pypto_pro.language as pl


@pl.jit()
def dump_data_tensor_full_kernel(
    out: pl.Tensor[[16], pl.DT_INT32],
):
    with pl.section_vector():
        for i in pl.range(0, 16):
            out[i] = i * 10
        pl.dump_data(out)
```

全量dump输出示例：

```text
[AIV Block 0/1] [dump_tensor] Data Type: int32, Layout: ND, Source Shape: [16], Offsets: [0], Shape: [16]
[AIV Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=int32, position=GM, dump_size=16
[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
```

动态偏移示例：

```python
# 循环变量作为偏移（动态）
for i in pl.range(0, 16, 4):
    pl.dump_data(out, offsets=[i], shapes=[4])

# get_block_idx() 作为偏移（动态）
vidx = pl.get_block_idx()
pl.dump_data(out, offsets=[vidx * 4], shapes=[4])
```

带标签输出（用于区分多个dump点）：

```python
pl.dump_data(out, flag="checkpoint_A")
```

标签输出示例：

```text
[AIV Block 0/1] [flag] checkpoint_A
[AIV Block 0/1] [dump_tensor] Data Type: int32, Layout: ND, Source Shape: [16], Offsets: [0], Shape: [16]
[AIV Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=int32, position=GM, dump_size=16
[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
```

### UB Tile输入

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_full_kernel(
    a: pl.Tensor[[8, 8], pl.DT_INT32],
    b: pl.Tensor[[8, 8], pl.DT_INT32],
    out: pl.Tensor[[8, 8], pl.DT_INT32],
):
    tt = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc)
        pl.store(out, tc, [0, 0])
```

全量dump输出示例：

```text
[AIV Block 0/1] [dump_tile] Data Type: int32, Layout: ND, Memory Space: UB, Source Shape: [8, 8], Offsets: [0, 0], Shape: [8, 8], Valid Shape: [8, 8]
[AIV Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=int32, position=UB, dump_size=64
[[0,2,4,6,8,10,12,14],
[16,18,20,22,24,26,28,30],
...,
[112,114,116,118,120,122,124,126]]
```

### Cube侧Tile输入

以下示例覆盖L1和L0C Buffer中的Tile打印。

全量dump：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_cube_tiles_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)

        pl.dump_data(cur_a, flag="l1")
        pl.dump_data(ac, flag="l0c")

        # 窗口打印同样适用于L1和L0C Tile
        pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8])

        pl.store(out, ac, [0, 0])
```

窗口模式输出示例（offsets=[16, 16], shapes=[8, 8]）：

```text
[AIC Block 0/1] [dump_tile] Data Type: float32, Layout: NZ, Memory Space: L0C Buffer, Source Shape: [64, 64], Offsets: [16, 16], Shape: [8, 8], Valid Shape: [8, 8]
[AIC Block 0] DumpTensor: desc=<dump_id>, addr=0x<address>, data_type=float32, position=L0C, dump_size=64
[[<value_0>,<value_1>,...,<value_7>],
...,
[<value_56>,<value_57>,...,<value_63>]]
```

在分块循环中使用窗口dump：

```python
for i in pl.range(0, 256, 64):
    for j in pl.range(0, 256, 64):
        # ... matmul ...
        pl.dump_data(ac, offsets=[8, 8], shapes=[32, 32])
        pl.store(out, ac, [i, j])
```
