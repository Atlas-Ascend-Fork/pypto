# pypto_pro.language.set_mask_count

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

将向量掩码寄存器切换为**count（计数）模式**。在该模式下，掩码寄存器的低64位被解释为参与运算的元素个数，而非逐位掩码。可直接设置计算数据量，实际迭代次数由Vector计算单元推断。

与[pypto_pro.language.set_vec_mask](set_vec_mask.md)配合使用：先切换到count模式，再设置元素个数，即可精确控制尾块计算的有效元素数。

## 函数原型

```python
pypto_pro.language.set_mask_count() -> None
```

## 参数说明

无。

## 返回值说明

无。

## 约束说明

- 切换到count模式后，须通过[pypto_pro.language.set_vec_mask](set_vec_mask.md)设置有效元素个数，否则掩码值不确定。
- count模式下，[pypto_pro.language.set_vec_mask](set_vec_mask.md)的mask_low参数被解释为元素个数（而非位掩码），mask_high忽略。
- count模式处理完尾块后，须先调用[pypto_pro.language.set_mask_norm](set_mask_norm.md)切回norm模式，再调用[pypto_pro.language.reset_mask](reset_mask.md)恢复全1掩码；pypto_pro.language.set_mask_norm本身不会改写当前掩码值。否则后续矢量计算的掩码行为不确定。
- Tile级操作（如[add](../elementwise/add.md)、[mul](../elementwise/mul.md)）内部会自动管理掩码，用户手动设置的count模式会被覆盖。pypto_pro.language.set_mask_count主要用于自定义操作或底层控制场景。

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
