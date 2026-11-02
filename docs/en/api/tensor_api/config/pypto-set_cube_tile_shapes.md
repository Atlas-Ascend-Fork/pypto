# pypto.set_cube_tile_shapes

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:31:19.151Z pushedAt=2026-10-28T11:57:29.591Z nl=2 -->

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

## Description

Before calling `pypto.matmul` or `pypto.scaled_mm`, you must call this API to set the tile shape for matrix operations. For details about the tile configuration, see [Matmul High-Performance Programming](../../../guide/programming_guide/tensor/debug/matmul_performance_guide.md).

## Prototype

```python
set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], enable_split_k: bool = False) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|-------------------|-----------|--------------------------------------------------------------------------------------------------|
| m | Input | Size of the tile shape on the m dimension at L0 and L1, which is a list of two integers corresponding to the sizes of mL0 and mL1. |
| k | Input | Size of the tile shape on the k dimension at L0 and L1, which is a list of two integers corresponding to the sizes of kL0 and kL1. |
| n | Input | Size of the tile shape on the n dimension at L0 and L1, which is a list of two integers corresponding to the sizes of nL0 and nL1. |
| enable_split_k | Input | Whether to enable the multi-core K-splitting feature of matmul. Set to True to enable it (a convenience switch that **does not guarantee optimal performance**); the default is False, indicating that multi-core K-splitting is not enabled.<br>For performance tuning, it is recommended to implement K-splitting manually on the frontend. For details, see [Matmul High-Performance Programming](../../../guide/programming_guide/tensor/debug/matmul_performance_guide.md). |

## Return Value

void

## Constraints

- Alignment constraints

    - General alignment constraints

    **mL0**, **mL1**, **kL0**, **kL1**, **nL0**, and **nL1** must all satisfy 32-byte alignment (the **DT_FP32** input scenario requires 16-element alignment). For example, when the input matrix data type is **DT_FP16**, `kL0 * sizeof(DT_FP16) % 32 == 0`.

    - Basic relationship constraints

    | Constraint Item | Requirement |
    |:-------|:-----|
    | mL0 and mL1 | `mL0 > 0` and `mL0 ≤ mL1` and `mL1 % mL0 == 0` |
    | kL0 and kL1 | `kL0 > 0` and `kL0 ≤ kL1` and `kL1 % kL0 == 0` |
    | nL0 and nL1 | `nL0 > 0` and `nL0 ≤ nL1` and `nL1 % nL0 == 0` |

    - ND-format-specific constraints

    When matrix A is in the ND **format** and transposed (that is, the data layout is [K, M]), mL0 must satisfy 32-byte alignment.

    - NZ-format-specific constraints

    When matrices A and B are in the NZ **format**, the outer-axis tile shape must satisfy 16-element alignment, and the inner-axis tile shape must satisfy 32-byte alignment. For example, when matrix A is not transposed, the outer axis is M and the inner axis is K, so mL0 and mL1 must satisfy 16-element alignment, and kL0 and kL1 must satisfy 32-byte alignment.

    - **scaled_mm** specific constraints

    When calling `pypto.scaled_mm`, the requirement `kL0 % 64 == 0` must be satisfied.

- Spatial constraint

    - When the input dtype is **DT_FP16**, **DT_BF16**, or **DT_FP32**:

    ```txt
    CeilAlign(mL0, 16) × CeilAlign(kL0, 16) × sizeof(aDtype) ≤ L0A_size
    CeilAlign(nL0, 16) × CeilAlign(kL0, 16) × sizeof(bDtype) ≤ L0B_size
    CeilAlign(mL0, 16) × CeilAlign(nL0, 16) × sizeof(cDtype) ≤ L0C_size
    CeilAlign(mL1, 16) × CeilAlign(kL1, 16) × sizeof(aDtype) + CeilAlign(nL1, 16) × CeilAlign(kL1, 16) × sizeof(bDtype) ≤ L1_size
    ```

    - When the input dtype is **DT_INT8**, **DT_FP8E5M2**, **DT_FP8E4M3**, or **DT_HF8**:

    ```txt
    CeilAlign(mL0, 32) × CeilAlign(kL0, 32) × sizeof(aDtype) ≤ L0A_size
    CeilAlign(nL0, 32) × CeilAlign(kL0, 32) × sizeof(bDtype) ≤ L0B_size
    CeilAlign(mL0, 32) × CeilAlign(nL0, 32) × sizeof(cDtype) ≤ L0C_size
    CeilAlign(mL1, 32) × CeilAlign(kL1, 32) × sizeof(aDtype) + CeilAlign(nL1, 32) × CeilAlign(kL1, 32) × sizeof(bDtype) ≤ L1_size
    ```

    - **Bias** spatial constraint:

    After the bias data reaches the BTBuffer, it is all converted to fp32 and must satisfy the following constraints:

    ```txt
    nL0 × 4 ≤ BTBuffer_size
    ```

    - **FixPipe** spatial constraint:

    The scaleTensor data is of type uint64_t and must satisfy the following constraints:

    ```txt
    nL0 × 8 ≤ FixBuffer_size
    ```

    where:
    - **aDtype** and **bDtype** are the input matrix data types.
    - **cDtype** is the output matrix data type. When the input is **DT_INT8**, **cDtype** is **DT_INT32**; in other scenarios, **cDtype** is **DT_FP32**.
    - The element alignment of `CeilAlign(value, align)` is implemented as `(value + align - 1) / align * align`.

- Multi-core K-splitting constraints.

    - `pypto.matmul` supports multi-core K-splitting for 2D/3D/4D matrices.
    - `pypto.scaled_mm` supports multi-core K-splitting only for 2D matrices, not for 3D/4D matrices.
    - In the multi-core K-splitting scenario, only **out\_dtype** data types **DT\_FP32** or **DT\_INT32** are supported.
    - The Bias/FixPipe (including ReLU) scenario does not support stacking the multi-core K-splitting feature.
    - Only the scenario where the tensor is created through the `pypto.view()` API is supported, and the valid_shape of the K axis must be exactly the same as the shape of the K axis. For details about the API, see [pypto.view](../operation/pypto-view.md).

## Examples

```python
# Basic configuration.
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# Enable multi-core K-splitting (a convenience switch that does not guarantee optimal performance; for performance tuning, see Matmul High-Performance Programming).
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
