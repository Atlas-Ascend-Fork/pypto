# pypto.set_cube_tile_shapes

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id7 -->
- Kirin 9050 processors: Supported
<!-- end id7 -->

## Description

Before calling `pypto.matmul` or `pypto.scaled_mm`, you must call this API to set the split size for matrix operations. For details about the split configuration, see [Matmul High-Performance Programming](../../../guide/programming_guide/tensor/debug/matmul_performance_guide.md).

## Prototype

```python
set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], enable_split_k: bool = False) -> None
```

## Parameters

| Parameter           | Input/Output| Description                                                                                              |
|-------------------|-----------|--------------------------------------------------------------------------------------------------|
| m                      | Input     | Tiling size of the TileShape along the M dimension in the L0 Buffer and L1 Buffer. The value is a list containing two integers, corresponding to the tiling size of mL0 and mL1, respectively.|
| k                      | Input     | Tiling size of the TileShape along the K dimension in the L0 Buffer and L1 Buffer. The value is a list containing two integers, corresponding to the tiling size of kL0 and kL1, respectively.|
| n                      | Input     | Tiling size of the TileShape along the N dimension in the L0 Buffer and L1 Buffer. The value is a list containing two integers, corresponding to the tiling size of nL0 and nL1, respectively.|
| enable_split_k         | Input     | If this parameter is set to True, the multi-core K-tiling function of matmul is enabled (this is a convenient switch and **does not guarantee the optimal performance**). The default value is False, indicating that the multi-core K-tiling function is disabled.<br>You are advised to manually perform K-tiling at the frontend for performance tuning. For details, see [Matmul High-Performance Programming](../../../guide/programming_guide/tensor/debug/matmul_performance_guide.md).|

## Return Value

void

## Constraints

- Alignment constraints

    - General alignment constraints

    mL0, mL1, kL0, kL1, nL0, and nL1 must be 32-byte aligned. (In the DT_FP32 input scenario, 16-element alignment is required.) For example, when the data type of the input matrix is DT_FP16, kL0 * sizeof(DT_FP16) % 32 == 0 must be satisfied.

    - Basic relationship constraints

    | Item| Requirement|
    |:-------|:-----|
    | mL0 and mL1| `mL0 > 0`, `mL0 ≤ mL1`, and `mL1 % mL0 == 0`|
    | kL0 and kL1| `kL0 > 0`, `kL0 ≤ kL1`, and `kL1 % kL0 == 0`|
    | nL0 and nL1| `nL0 > 0`, `nL0 ≤ nL1`, and `nL1 % nL0 == 0`|

    - Constraints specific to the ND format

    When matrix A is in ND format and transposed (the data layout is [K, M]), mL0 must be 32-byte aligned.

    - Constraints specific to the NZ format

    When matrices A and B are in NZ format, the outer-axis tilting size must be 16-element aligned, and the inner-axis tilting size must be 32-byte aligned. Example: In the non-transposed scenario of matrix A, the outer axis is M and the inner axis is K. mL0 and mL1 must be 16-element aligned, and kL0 and kL1 must be 32-byte aligned.

    - Constraints specific to scaled_mm

    When `pypto.scaled_mm` is called, `kL0 % 64 == 0` must be satisfied.

- Space constraints

    - If the input dtype is DT_FP16, DT_BF16, or DT_FP32:

    ```txt
    CeilAlign(mL0, 16) × CeilAlign(kL0, 16) × sizeof(aDtype) ≤ L0A_size
    CeilAlign(nL0, 16) × CeilAlign(kL0, 16) × sizeof(bDtype) ≤ L0B_size
    CeilAlign(mL0, 16) × CeilAlign(nL0, 16) × sizeof(cDtype) ≤ L0C_size
    CeilAlign(mL1, 16) × CeilAlign(kL1, 16) × sizeof(aDtype) + CeilAlign(nL1, 16) × CeilAlign(kL1, 16) × sizeof(bDtype) ≤ L1_size
    ```

    - If the input dtype is DT_INT8, DT_FP8E5M2, DT_FP8E4M3, or DT_HF8:

    ```txt
    CeilAlign(mL0, 32) × CeilAlign(kL0, 32) × sizeof(aDtype) ≤ L0A_size
    CeilAlign(nL0, 32) × CeilAlign(kL0, 32) × sizeof(bDtype) ≤ L0B_size
    CeilAlign(mL0, 32) × CeilAlign(nL0, 32) × sizeof(cDtype) ≤ L0C_size
    CeilAlign(mL1, 32) × CeilAlign(kL1, 32) × sizeof(aDtype) + CeilAlign(nL1, 32) × CeilAlign(kL1, 32) × sizeof(bDtype) ≤ L1_size
    ```

    - Bias space constraints:

    When transferred to the BTBuffer, the bias data is converted to fp32. The following constraints must be satisfied:

    ```txt
    nL0 × 4 ≤ BTBuffer_size
    ```

    - FixPipe space constraints:

    The scaleTensor data is uint64_t. The following constraints must be satisfied:

    ```txt
    nL0 × 8 ≤ FixBuffer_size
    ```

    Where:
    - aDtype and bDtype are the data types of the input matrices.
    - cDtype is the data type of the output matrix. When the input is DT_INT8, cDtype is DT_INT32. In other scenarios, cDtype is DT_FP32.
    - The `CeilAlign(value, align)` element alignment is implemented as follows: `(value + align - 1) / align * align`

- Constraints on multi-core K-tiling

    <!-- npu="x90,9030,9050" id6 -->
    - Kirin 9030, Kirin X90, and Kirin 9050 processors do not support multi-core K-tiling (`enable_split_k`).
    <!-- end id6 -->
    - `pypto.matmul` supports multi-core K-tiling for 2D, 3D, and 4D matrices.
    - `pypto.scaled_mm` supports multi-core K-tiling only for 2D matrices, and does not support multi-core K-tiling for 3D or 4D matrices.
    - In the multi-core K-tiling scenario, out\_dtype can only be DT\_FP32 or DT\_INT32.
    - The multi-core K-tiling function cannot be used together with the Bias/FixPipe (including ReLU) function.
    - This function is only supported in scenarios where tensors are created using the `pypto.view()` API, and valid_shape of the K axis must be the same as the shape of the K axis. For details about how to use the API, see [pypto.view](../operation/pypto-view.md).

## Examples

```python
# Basic configuration
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# Enable multi-core K-tiling. (This is a convenient switch and does not guarantee the optimal performance. For details about performance tuning, see "Matmul High-Performance Programming".)
m = 1024
n = 1024
k = 1024
a_tensor = pypto.tensor((m, k), pypto.DT_FP16, "a_tensor")
b_tensor = pypto.tensor((k, n), pypto.DT_FP16, "b_tensor")
m_view = 256
m_loop = m // m_view
pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128], enable_split_k=True)
for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
    m_offset = m_idx * m_view
    a_view = pypto.view(a_tensor, [m_view, k],
                                              [m_idx * m_view, 0],
                                              valid_shape=[(m - m_offset).min(m_view), k])
    pypto.matmul(a_view, b_tensor, pypto.DT_FP32)
```
