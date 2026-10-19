# Tensor创建和操作

Tensor用于描述GM中的多维数据，作为Kernel的输入、输出或由裸指针构造的数据视图。Tensor记录数据类型、shape、stride和layout等信息，但不负责申请GM内存。Kernel计算使用的片上缓冲区由Tile表示，相关内容请参考[Tile创建和操作](tile_creation_and_operations.md)。

## 在Kernel签名中声明Tensor

Host侧使用PyTorch在NPU上创建输入和输出数据，Kernel通过pypto_pro.language.Tensor类型标注接收这些数据。Tensor类型标注的基本形式如下：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def kernel(
    x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
):
    ...
```

Tensor[[shape...], dtype]中的第一个参数是shape，第二个参数是元素数据类型。还可以使用第三个参数声明GM数据的layout：

```python
import pypto_pro.language as pl


nz_tensor: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]
```

layout标注只描述GM中已有数据的排布，不执行不同layout之间的数据转换。数据转换由支持相应格式的搬运接口完成。

### 输入和输出方向标注

使用Profiling采集Kernel信息时，可以通过Tensor类型标注中的pypto_pro.language.Input或pypto_pro.language.Output枚举值声明参数方向：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def kernel(
    x: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.Input],
    out: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16, pl.Output],
):
    ...
```

pypto_pro.language.Input表示输入，pypto_pro.language.Output表示输出。direction可省略，未标注时默认使用pypto_pro.language.Input；需要在Profiling结果中将Tensor上报为输出时，使用pypto_pro.language.Output。

在方括号形式中，direction可与layout、memref一起声明，推荐将direction放在标注末尾，例如`pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ, pl.Output]`。调用式也支持将direction作为第三个参数，例如`pl.Tensor([64, 128], pl.DT_FP16, pl.Output)`。

方向标注只影响Profiling结果中的Tensor输入、输出信息，不改变Kernel的读写语义。完整语法和约束请参考[pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md)。

### Shape声明方式

Tensor的每一维可以使用以下方式声明：

| 声明方式 | 含义 | 对编译变体的影响 |
|:---|:---|:---|
| 正整数 | 固定维度，启动时实际尺寸必须与声明一致 | 尺寸固定 |
| pypto_pro.language.DYNAMIC | 运行时动态维度 | 维度值变化时复用同一编译变体 |
| pypto_pro.language.STATIC | 编译期特化维度 | 维度值变化时生成新的编译变体 |
| 末尾的... | 展开剩余维度，各维均按STATIC处理 | rank或维度值变化时生成新的编译变体 |

不同方式可以混合使用：

```python
import pypto_pro.language as pl


# 固定、动态和编译期特化维度混合声明。
x: pl.Tensor[[64, pl.DYNAMIC, pl.STATIC], pl.DT_FP16]

# rank在调用时确定；第一个维度为DYNAMIC，其余维度按STATIC处理。
y: pl.Tensor[[pl.DYNAMIC, ...], pl.DT_FP16]
```

省略号最多出现一次且必须位于shape末尾。Tensor的完整类型约束请参考[pypto_pro.language.Tensor](../../../../api/pro_api/SIMD-API/basic_data_structures/Tensor.md)。

## 从Ptr或Tensor创建Tensor视图

当Kernel接收裸指针或需要用新的shape、stride解释已有Tensor时，可以调用pypto_pro.language.make_tensor创建Tensor视图：

```python
pypto_pro.language.make_tensor(
    ptr,
    shape,
    stride=None,
    dtype=None,
)
```

ptr可以是pypto_pro.language.Ptr或已有Tensor。新Tensor与源对象共享同一段GM地址；make_tensor不申请内存，也不复制或重排数据。

make_tensor还支持`depth`、`fwd_ids`、`bwd_ids`三个可选参数，用于把一段workspace声明为Cube与Vector之间的跨核共享Buffer，仅在[自动CV并行流水](../advanced_programming/auto_parallel_pipeline.md)场景下生效，未开启自动流水时忽略。

### 创建连续Tensor视图

省略stride时，框架根据shape生成连续的行主序stride：

```python
import pypto_pro.language as pl


tensor = pl.make_tensor(ptr, [8, 16])
# 等价于：tensor = pl.make_tensor(ptr, [8, 16], [16, 1])
```

shape中的维度既可以是编译期整数，也可以是Kernel中的运行时整型Scalar表达式。因此，动态shape也可以通过TilingData传入后用于创建固定rank的Tensor视图：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dynamic_kernel(
    x: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
    tiling: OpTiling,
):
    tensor_x = pl.make_tensor(x, [tiling.m, tiling.n])
    tensor_out = pl.make_tensor(out, [tiling.m, tiling.n])
    ...
```

TilingData的声明和传入方式请参考[Tiling参数定义与传递](tiling/tiling_parameter_definition.md#tilingdata)。

### 创建带显式stride的Tensor视图

显式传入stride可以描述行间不连续或轴交换后的GM视图。stride包含的元素个数必须与shape的维数相同，单位是元素，不是字节。对于不足8 bit的数据类型，最后一维stride必须是编译期常量1：

```python
import pypto_pro.language as pl


# 每行包含16个连续元素，相邻两行的起始位置相隔32个元素。
pitched = pl.make_tensor(ptr, [8, 16], [32, 1])

# normal[i, j]与transposed[j, i]指向相同地址。
normal = pl.make_tensor(ptr, [8, 16], [16, 1])
transposed = pl.make_tensor(ptr, [16, 8], [1, 16])
```

上述操作只改变逻辑索引到GM地址的映射，不会转置或搬运原数据。调用方需要保证shape、stride和dtype描述的访问范围没有超出源内存。详细参数说明请参考[pypto_pro.language.make_tensor](../../../../api/pro_api/SIMD-API/resource_management/make_tensor.md)。

## 读取Tensor的Shape

Kernel内通过tensor.shape[axis]读取Tensor维度。axis必须是编译期整数，支持负索引。读取到的维度可以参与地址计算、循环边界和多核切分：

```python
m = x.shape[0]
n = x.shape[-1]
tile_rows = (m + TILE_M - 1) // TILE_M
tile_cols = (n + TILE_N - 1) // TILE_N
```

对于DYNAMIC维度，shape值在运行时取得；对于STATIC维度，shape值会固化到相应编译变体中。

## Tensor别名和指针转换

### 创建Tensor别名

通过Python赋值可以为Tensor创建别名。别名与原Tensor指向同一段GM内存，不产生数据复制：

```python
original_input = input_tensor
original_input_alias = original_input

# 重新绑定原变量不会改变已有别名的指向。
input_tensor = replacement_tensor
```

### 从Tensor获取Ptr

pypto_pro.language.make_ptr可以从Tensor提取底层指针，也可以为已有Ptr创建新的元素类型视图：

```python
import pypto_pro.language as pl


ptr = pl.make_ptr(tensor)
fp16_ptr = pl.make_ptr(byte_ptr, dtype=pl.DT_FP16)
```

返回的Ptr与源对象共享地址。指定dtype只改变地址的元素类型解释，不会转换原数据；调用方需要保证地址对齐和可访问范围正确。详细说明请参考[pypto_pro.language.make_ptr](../../../../api/pro_api/SIMD-API/resource_management/make_ptr.md)。

需要对Ptr按元素进行偏移时，可以使用[pypto_pro.language.addptr](../../../../api/pro_api/SIMD-API/resource_management/addptr.md)，再通过make_tensor将偏移后的地址包装为Tensor视图。addptr不支持不足8 bit的数据类型，需先通过make_ptr重解释为DT_UINT8并按字节偏移。这种方式常用于将一块GM Workspace划分为多个区域。

## Tensor与Tile之间的数据搬运

Tensor位于GM，不能直接作为Tile计算接口的片上操作数。Kernel通常先将Tensor中的数据搬入Tile，完成计算后再将结果搬回Tensor：

```python
import pypto_pro.language as pl


with pl.section_vector():
    pl.load(input_tile, input_tensor, [row_offset, col_offset])
    # Tile计算过程省略。
    pl.store(output_tensor, output_tile, [row_offset, col_offset])
```

- load和store使用Tensor中的元素坐标定位搬运起点。
- load_tile和store_tile按Tile网格坐标定位数据块。
- Tensor不支持切片子视图（例如`tensor[0:32, :]`）。访问其中一部分数据时，通过load和store的元素坐标定位，或先用addptr偏移地址再用make_tensor创建新视图。
- 搬运范围、stride、layout和尾块受具体搬运接口约束；编译期可确定的非法参数由框架检查，动态访问范围由调用方保证。

Tensor只描述GM数据视图；Tile的创建、片上地址和缓冲区管理请参考[Tile创建和操作](tile_creation_and_operations.md)，Tile上的矢量计算请参考[Tile计算](vector_computation/tile_computation.md)，矩阵计算请参考[Cube计算](cube_computation.md)。
