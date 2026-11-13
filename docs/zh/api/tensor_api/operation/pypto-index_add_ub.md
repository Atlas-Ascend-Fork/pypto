# pypto.index\_add\_ub

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

pypto.index_add__ub的non-inplace版本，可参考[pypto.index_add__ub](pypto-index_add__ub.md)。

## 函数原型

```python
index_add_ub(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## 参数说明

请参考[pypto.index_add__ub](pypto-index_add__ub.md)的参数说明。

## 返回值说明

请参考[pypto.index_add__ub](pypto-index_add__ub.md)的返回值说明。

## 约束说明

- 请参考[pypto.index_add__ub](pypto-index_add__ub.md)的约束说明。
- Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。
如输入input为[m, n, p]，dim为1，输入source为[m, t, p]，输入index为[t]，输出为[m, n, p]，TileShape设置为[m1, t1, p1]，则m1，p1分别用于切分m，p轴。n轴，t轴不可切，必须保证n轴t轴全载。

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
source = pypto.tensor([3, 3], pypto.DT_INT32)   # shape (3, 3)
index = pypto.tensor([3], pypto.DT_INT32)   # shape (3,)
dim = 0
# use alpha
y = pypto.index_add_ub(x, dim, index, source, alpha=1)
# not use alpha
y = pypto.index_add_ub(x, dim, index, source)
```

结果示例如下：

```python
输入数据x:   [[0 0 0],
               [0 0 0]]
      source: [[1 1 1],
               [1 1 1],
               [1 1 1]]
      index:   [0 1 0]
输出数据y:   [[2 2 2],
               [1 1 1]]               # shape (2, 3)
```
