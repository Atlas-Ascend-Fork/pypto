# pypto_pro.language.set_mask_norm

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

将向量掩码寄存器切换回**norm（普通位掩码）模式**。在该模式下，掩码寄存器的128位被解释为逐位掩码，每一位对应一个元素是否参与运算。

与[pypto_pro.language.set_mask_count](set_mask_count.md)互为反向操作：count模式用于尾块计数，norm模式恢复默认的逐位掩码。

## 函数原型

```python
pypto_pro.language.set_mask_norm() -> None
```

## 参数说明

无。

## 返回值说明

无。

## 约束说明

- pypto_pro.language.set_mask_norm只切换掩码的解释模式，不会改写当前掩码值。
- 在count模式处理完尾块后，须调用pypto_pro.language.set_mask_norm切回norm模式，否则后续矢量计算的掩码行为不确定。
- 切回norm模式后，掩码寄存器仍保持count模式下设置的值，须通过[pypto_pro.language.set_vec_mask](set_vec_mask.md)重新设置或调用[pypto_pro.language.reset_mask](reset_mask.md)恢复全1掩码。
- Tile级操作（如[add](../elementwise/add.md)、[mul](../elementwise/mul.md)）内部会自动管理掩码，用户手动设置的模式会被覆盖。pypto_pro.language.set_mask_norm主要用于自定义操作或底层控制场景。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mask_count_norm_kernel(
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
        pl.set_mask_count()
        pl.set_vec_mask(0, 64 * 64)
        # 在此执行读取count模式掩码的自定义矢量计算
        pl.set_mask_norm()
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
