# pypto.get\_conv\_tile\_shapes

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="9030" id4 -->
- Kirin9030处理器系列产品：不支持
<!-- end id4 -->
<!-- npu="x90" id5 -->
- KirinX90处理器系列产品：不支持
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin9050处理器系列产品：不支持
<!-- end id6 -->

## 功能说明

获取卷积（conv）计算中设置的TileShape大小以及L0TileInfo的开关启用。

## 函数原型

```python
get_conv_tile_shapes() -> Tuple[pypto_impl.TileL1Info, pypto_impl.TileL0Info, bool]
```

## 参数说明

无。

## 返回值说明

返回L0和L1 Buffer上的TileShape大小、是否开启L0TileInfo的开关。

## 约束说明

无。

## 调用示例

```python
pypto.get_conv_tile_shapes()
```
