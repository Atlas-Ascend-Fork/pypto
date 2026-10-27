# pypto_pro.language.reset_mask

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

重置向量掩码寄存器为默认状态（全1），使后续矢量计算的所有元素均参与运算。

等价于pypto_pro.language.set_vec_mask(-1, -1)，即128位掩码全部置1。

## 函数原型

```python
pypto_pro.language.reset_mask() -> None
```

## 参数说明

无。

## 返回值说明

无。

## 约束说明

- 调用[pypto_pro.language.set_vec_mask](set_vec_mask.md)设置掩码后，如需恢复默认值（全1），可调用本接口。
- 若当前处于count模式，pypto_pro.language.reset_mask恢复全1掩码但不会自动切回norm模式。count模式下全1掩码的语义与norm模式不同，建议先调用[pypto_pro.language.set_mask_norm](set_mask_norm.md)切回norm模式，再调用pypto_pro.language.reset_mask。
- Tile级操作（如[add](../elementwise/add.md)、[mul](../elementwise/mul.md)）内部会自动管理掩码，用户通过pypto_pro.language.set_vec_mask设置的掩码会被覆盖。pypto_pro.language.reset_mask主要用于自定义操作或底层控制场景。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def reset_mask_kernel(
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
        pl.set_vec_mask(0, 0xFFFFFFFF)
        # 在此执行读取当前掩码状态的自定义矢量计算
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
