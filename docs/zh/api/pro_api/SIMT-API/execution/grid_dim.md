# pypto_pro.language.simt.grid_dim

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

获取当前线程块网格在X、Y、Z三个维度上的线程块数量。

## 函数原型

```python
pypto_pro.language.simt.grid_dim() -> Any
```

## 参数说明

无。

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回三维线程块网格尺寸元组对象，通过grid.x、grid.y和grid.z读取各维大小，每个分量均为DT_UINT32类型的Scalar。

- SIMT编程：返回Host传入的grid，例如grid=(2, 3, 4)时，返回(2, 3, 4)。
- SIMD与SIMT混合编程：grid.x为外层Kernel实际生效的逻辑Vector（AIV）核数，grid.y和grid.z均为1。实际生效值与Host传入的block_dim的关系参见[blockDim的含义与设置](../../../../guide/programming_guide/pro/development/kernel_function.md#blockdim的含义与设置)。

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_grid_dim_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    grid = pl.simt.grid_dim()
    output[0, tid] = grid.x


@pl.jit()
def simt_grid_dim_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_grid_dim_x[256](output)
```
