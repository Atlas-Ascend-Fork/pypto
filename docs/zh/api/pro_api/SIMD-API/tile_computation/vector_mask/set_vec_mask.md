# pypto_pro.language.set_vec_mask

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

显式设置向量掩码寄存器的128位值（高64位 + 低64位），按位精确指定哪些元素参与后续矢量计算。

掩码寄存器的含义取决于当前模式：

- **norm模式**（默认）：128位逐位掩码，每一位对应一个元素是否活跃。bit值为1表示参与计算，0表示不参与。
- **count模式**：mask_low被解释为有效元素个数，mask_high忽略。

模式切换通过[pypto_pro.language.set_mask_norm](set_mask_norm.md) / [pypto_pro.language.set_mask_count](set_mask_count.md)控制。

## 函数原型

```python
pypto_pro.language.set_vec_mask(mask_high: int, mask_low: int) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| mask_high | 输入 | 表示64位掩码位模式的整型常量或运行时整型标量表达式。norm模式下控制第64-127号元素的活跃状态；count模式下忽略。 |
| mask_low | 输入 | 表示64位掩码位模式的整型常量或运行时整型标量表达式。norm模式下控制第0-63号元素的活跃状态；count模式下为有效元素个数。 |

## 返回值说明

无。

## 约束说明

- norm模式下，mask_high和mask_low是高、低各64位的原始掩码值。有效bit数及bit与数据元素的对应关系由后续矢量指令的数据类型和指令语义决定。
- count模式下，mask_low表示参与计算的总元素个数，不按位解释。该模式用于处理尾块（有效元素数非对齐场景），实际迭代次数由Vector计算单元自动推断。
- norm模式下将高、低64位均设为0，表示后续使用该掩码的矢量指令没有活跃元素；pypto_pro.language.set_vec_mask调用本身仍会写掩码寄存器。
- Tile级操作（如[add](../elementwise/add.md)、[mul](../elementwise/mul.md)）内部会自动管理掩码，用户通过pypto_pro.language.set_vec_mask设置的掩码会被覆盖。pypto_pro.language.set_vec_mask主要用于自定义操作或底层控制场景。
- 使用pypto_pro.language.set_vec_mask设置掩码后，须在后续计算完成后调用[pypto_pro.language.reset_mask](reset_mask.md)恢复默认全1掩码，否则可能影响后续矢量计算。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def set_vec_mask_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        # 高64位清零，低64位的低32个掩码位设为1
        pl.set_vec_mask(0, 0xFFFFFFFF)
        # 在此执行读取当前掩码状态的自定义矢量计算
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
