# Matmul Performance Tuning Guide

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T02:18:37.481Z pushedAt=2026-10-27T09:27:30.075Z nl=2 -->

PyPTO provides users with an efficient and convenient operator development framework. In terms of performance optimization, PyPTO offers a rich set of configurable options, including operator tile configuration, algorithm configuration, and graph optimization configuration. These options provide users with great development flexibility on one hand, while significantly raising the barrier to entry on the other.

This tutorial aims to provide users with a methodology for analyzing and optimizing Matmul operators, helping them achieve better overall performance when developing operators with PyPTO.

## Performance Optimization Objectives

Analyzing the theoretical performance upper bound is the first step in operator performance optimization. For any operator or algorithm, the process of completing a certain computation generally involves two major parts: data movement and data computation. The industry commonly uses arithmetic intensity to measure the proportional relationship between these two parts, which also serves as an important basis for understanding the theoretical performance upper bound of an operator.

Arithmetic intensity reflects the proportional relationship between the amount of computation and the amount of data access for an operator or algorithm at the objective theoretical level. It is defined as the total number of floating-point operations (FLOPs) divided by the total number of memory access bytes, with the unit `FLOPs/Byte`. For a given hardware platform, the ratio of its peak compute capability to its peak bandwidth is defined as the compute-to-bandwidth ratio (which obviously has the same unit as arithmetic intensity). When the arithmetic intensity is greater than the compute-to-bandwidth ratio, the performance of the algorithm on that hardware platform is considered to be limited by compute capability (i.e., Compute Bound); otherwise, it is limited by memory bandwidth (i.e., Memory Bound).

The following further analyzes the performance upper bound of the Matmul operator on a given hardware platform.

Consider the data load volume (commonly referred to as MTE2 load volume) and computation volume of an ordinary cube multiplication. The condition for being Compute Bound is:
$
\frac{CP}{BW} \leq \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$
In the above formula, $CP$ and $BW$ represent the peak compute capability and bandwidth of the given hardware platform, and $mL1$ and $nL1$ represent the tile sizes along the M and N axes, which are the tile sizes that users need to specify when calling `pypto.set_cube_tile_shapes`.

*Note: The theoretical formula above only considers the most basic inner product algorithm and temporarily ignores the data write-out overhead to simplify the analysis.*

It can be seen that, on the one hand, the tile size directly determines the number of tasks after Matmul partitioning, thereby determining the number of occupied cores and computation rounds during execution. On the other hand, the tile size theoretically determines the arithmetic intensity of the Matmul operator.

Therefore, the first step in optimizing Matmul performance is to optimize the tile configuration.

## Tile Configuration Optimization

### Increasing Arithmetic Intensity

When M and N are sufficiently large (such as in training scenarios), the arithmetic intensity of Matmul before partitioning is high enough and the number of tasks after partitioning is sufficient to fully occupy all cores. In this case, we tend to choose larger tile configurations in both the M and N dimensions, thereby achieving the highest possible post-partitioning arithmetic intensity in the hope of reaching Compute Bound.

Intuitively, the larger the tile, the higher the arithmetic intensity, and the easier it is for Matmul computation to reach Compute Bound. This is because tiling inevitably introduces repeated data movement, and more tiling leads to a greater volume of repeated data movement, which in turn lowers the arithmetic intensity. On the other hand, the tile size cannot increase indefinitely due to the constraints of on-chip multi-level buffer space (L1, L0).

Denote the tile sizes as:

```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1], enable_split_k=False)
```

where mL0, kL0, and nL0 represent the tile sizes in the L0 buffer, and mL1, kL1, and nL1 represent the tile sizes in the L1 buffer.

Since the CUBE computation on the NPU uses fractal data blocks as the minimum computation granularity, the tile sizes must also meet the requirements of the fractal format (that is, the outer axis is aligned to 16 elements, and the inner axis is aligned to 32 bytes). In addition, the tile sizes must satisfy the buffer space constraints. For details about the specific computation constraints, see pypto.set_cube_tile_shapes.

Taking the Atlas A3 series and Atlas A2 series products as examples, for scenarios where both cubes A and B are of the FP16 type, the recommended tile configurations that satisfy the buffer space constraints are as follows:

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=False)
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128], enable_split_k=False)
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128], enable_split_k=False)
```

Advantages of the preceding tile configurations:

- A high arithmetic intensity can be achieved while satisfying the L0 buffer constraints.

$
AI = \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$
According to the formula above, AI obviously reaches its maximum when $mL1 = nL1$. And since $mL1 * nL1 * sizeof(float) <= L0C\_SIZE = 131072$, the maximum AI is achieved when $mL1 = nL1 = sqrt(L0C\_SIZE / sizeof(float)) = 181$. However, because the tile sizes must meet the alignment requirements of the fractal format, and the impact of tile sizes on write and read bandwidth must also be considered, a combination of 128–256 is generally used.

- Both MTE2 and MTE1 transfers can enable double buffer, allowing pipeline parallelism.
 Under the above configuration, the space occupied by L0A and L0B is 32 KB, which is just enough to enable the MTE1 double buffer. Similarly, under the above tile configuration, MTE2 can also enable double buffer. In addition, since kL1 > kL0 and large-packet transfer is enabled, the transfer amount of a single MTE2 can be further increased, which helps improve the bandwidth utilization of MTE2.
 However, note that when mL1 and nL1 use the 128-256 combination, nbuffer cannot be enabled on L0C. Therefore, this tile configuration is suitable for scenarios with a relatively large K dimension (that is, a relatively small number of move-out operations). When frequent move-out is required, consider using the 128-128 combination.

It should be particularly noted that the above tile configuration is not fixed and needs to be comprehensively considered based on the compute scenario (including input shape, dtype, format, etc.) and the hardware platform. In fact, if Matmul is viewed as an algorithm, then the tile configuration (Tiling) is the most critical part of this algorithm.

At the same time, achieving Compute Bound requires not only improving arithmetic intensity but also maximizing memory access bandwidth, which will be discussed in the following sections.

#### Reducing Repeated Loading

When either M or N is relatively small (such as in inference scenarios), the arithmetic intensity of Matmul before tiling is relatively low, and generally only Memory Bound can be achieved. In this case, our optimization approach is to minimize repeated data loading (primarily MTE2 repeated loading) while ensuring full core utilization.

Generally, we need to ensure that the number of occupied cores reaches at least 80% of the total cores (e.g., at least 20 cores on a 24-core platform) to guarantee high MTE2 bandwidth utilization. This is expressed as follows:
$
\frac{M}{mL1} \cdot \frac{N}{nL1} >= 0.8 \cdot coreNum
$
On this basis, the repeated load volume should be minimized as much as possible. When the K-axis has L1 tiling, the MTE2 total load volume is:
$
MTE2\_LOAD\_SIZE = M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte
$
In summary, the tile configuration problem at this point reduces to a constrained extremum problem.

For example, consider a scenario with input dimensions M = 96, K = 1536, N = 3072, and data type FP16.
To avoid MTE2 repeated loading of cube B, a good tiling approach is to set mL1 = M = 96. Additionally, to fully utilize all cores, it is recommended to set nL1 = N / coreNum = 128. On this basis, further considering MTE2 and MTE1 pipeline parallelism as well as MTE2 bandwidth utilization, set kL1 = 4kL0.
In summary, a good tile configuration is:

```python
pypto.set_cube_tile_shapes([96, 96], [64, 256], [128, 128], enable_split_k=False)
```

For further optimization, cube A can be loaded into L1 once and kept resident for repeated use. This further reduces the MTE2 total load volume to:
$
MTE2\_LOAD\_SIZE = M \cdot K \cdot aByte + K \cdot N \cdot bByte
$
This completely eliminates MTE2 repeated loading and further optimizes overall performance. The tile configuration in this case is:

```python
pypto.set_cube_tile_shapes([96, 96], [64, 1536, 256], [128, 128], enable_split_k=False)
```

A relatively advanced tile configuration approach is used here, which independently sets kAL1 and kBL1, as shown below:

```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1], enable_split_k=False)
```

Here, kAL1 and kBL1 respectively represent the tile sizes of cube A and cube B along the K dimension in L1. When only two tile values are set for the K dimension, it means that kAL1 = kBL1 = kL1.

#### K-Axis Core Splitting

For scenarios where M and N are small but the K-axis is large, partitioning only across the M and N axes may fail to fully occupy all cores, resulting in poor overall performance. In such cases, a K-axis core splitting strategy can be adopted for optimization.

There are two ways to enable multi-core K splitting:

- **Manual graph construction**: Manually split the K-axis and call `pypto.matmul` and `pypto.add` to construct the multi-core K splitting graph. This method allows independent configuration of the single-core split length and the Vector tile size, making it suitable for in-depth performance tuning scenarios.
- **Automatic graph construction**: Use the `enable_split_k` switch to enable multi-core K splitting. This method is suitable for quick verification scenarios, but **does not guarantee optimal performance**.

##### Manual Graph Construction

The following shows example code:

```python
@pypto.frontend.jit
def matmul_demo_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
    pypto.set_vec_tile_shapes(32, 256) # Configure the add tile size.

    partial_sum = []
    # Before use, define k_size (generally the K dimension value of cube a) and k_view_size (the size of the K-axis view shape) in advance.
    k_size = a.shape[1] # Assume that cube a is not transposed.
    k_view_size = 32
    k_loop = (k_size + k_view_size - 1) // k_view_size # Configure the accumulation length of a single core.
    for k_idx in range(k_loop):
        a_view = a[:, k_idx * k_view_size: k_idx * k_view_size + k_view_size]
        b_view = b[k_idx * k_view_size: k_idx * k_view_size + k_view_size, :]
        res = pypto.matmul(a_view, b_view, pypto.DT_FP32)
        partial_sum.append(res)

    for i in range(1, len(partial_sum)):
        partial_sum[0] += partial_sum[i]

    out[:] = partial_sum[0]
```

##### Automatic Graph Construction Mode

For quick verification scenarios, you can quickly enable K-axis core splitting by setting `enable_split_k=True`:

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
```

In this case, `kL1` is used by default to split the K-axis. Each core computes a partial sum of length `kL1` and then moves it out, and finally all partial sums are accumulated. The code implementation is as follows:

```python
@pypto.frontend.jit
def matmul_split_k_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=True)
    out[:] = pypto.matmul(a, b, pypto.DT_FP32)
```

## Memory Bandwidth Optimization

### Increasing L2 Hit Rate

Returning to the Compute Bound criterion formula:
$$
\frac{CP}{BW} \leq \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$$
The optimization measures described earlier mainly focus on increasing arithmetic intensity. This section focuses on how to improve bandwidth utilization to reduce the compute-to-bandwidth ratio. Considering the impact of the L2 hit rate on the overall bandwidth, for the pure inner product algorithm, the total MTE2 transfer amount is calculated as follows:
$$
MTE2\_TOTAL\_LOAD\_SIZE = M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte
$$
The amounts of data transferred from HBM and L2 are respectively:
$$
HBM\_LOAD\_SIZE = M \cdot K \cdot aByte + K \cdot N \cdot bByte
$$
$$
L2\_LOAD\_SIZE = MTE2\_TOTAL\_LOAD\_SIZE - HBM\_LOAD\_SIZE
$$
Let the L2 hit rate be the proportion of the L2 load amount in the total load amount:
$$
l2\_hit\_ratio = \frac{L2\_LOAD\_SIZE}{MTE2\_TOTAL\_LOAD\_SIZE}=1-\frac{\frac{1}{N} \cdot aByte+\frac{1}{M} \cdot bByte}{\frac{1}{nL1} \cdot aByte+\frac{1}{mL1} \cdot bByte}
$$
For Atlas A3 series products and Atlas A2 series products, the L2 bandwidth is more than three times the HBM bandwidth, so the L2 hit rate should be improved first.

Further considering the scenario where both A and B cube data types are FP16, and taking into account the single-round L2 hit rate, we have:
$
l2\_hit\_ratio = 1-\frac{\frac{1}{nDim \cdot nL1} +\frac{1}{mDim \cdot mL1} }{\frac{1}{nL1} +\frac{1}{mL1} }
$
In the above formula, $mDim and nDim$ are the numbers of occupied cores along the M and N axes in a single round of computation, respectively. For a given tile configuration, it is evident that the single-round L2 hit rate is maximized when $nDim \cdot nL1 = mDim \cdot mL1$.

Considering both arithmetic intensity and L2 hit rate, taking Atlas A3 series products and Atlas A2 series products as examples, to maximize arithmetic intensity, we generally use a tile configuration of 128–256. Taking mL1=128 and nL1=256 as an example, the L2 hit rate is the highest when the numbers of occupied cores along the M and N axes in a single round satisfy $nDim \cdot 256 = mDim \cdot 128$. For a 24-core platform, mDim=6, nDim=4 or mDim=8, nDim=3 can be used.

For example, for M = N = K = 6144 in the FP16 scenario, using only the 128-256 tile configuration, the overall latency is approximately 2.1 ms, with an equivalent compute power of about 220 TFLOPS. The test code is as follows:

```python
import torch
import pypto


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0})
def matmul_native_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
    out[:] = pypto.matmul(a, b, pypto.DT_FP16)


def test_native_mm():
    M, K, N = 6144, 6144, 6144
    a = torch.randn([M, K], dtype=torch.float16, device="npu:0")
    b = torch.randn([K, N], dtype=torch.float16, device="npu:0")
    out = torch.empty(M, N, dtype=torch.float16, device="npu:0")
    matmul_native_kernel(a, b, out)


if __name__ == "__main__":
    test_native_mm()
```

Based on the preceding tile configuration, the L2 hit rate is further optimized, reducing the overall latency to approximately 1.6 ms with an equivalent compute power of approximately 290 TFLOPS. The test code is as follows:

```python
import torch
import pypto


M, K, N = 6144, 6144, 6144
mL1, nL1 = 128, 256
mDim, nDim = 6, 4
m_view = mL1 * mDim   # 768
n_view = nL1 * nDim   # 1024


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0})
def matmul_l2_split_kernel(
    a: pypto.Tensor([], pypto.DT_FP16),
    b: pypto.Tensor([], pypto.DT_FP16),
    out: pypto.Tensor([], pypto.DT_FP16),
):
    pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])

    m_loop = (M + m_view - 1) // m_view
    n_loop = (N + n_view - 1) // n_view
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            a_view = a[m_idx * m_view : m_idx * m_view + m_view, :]
            b_view = b[:, n_idx * n_view : n_idx * n_view + n_view]
            out_view = pypto.matmul(a_view, b_view, pypto.DT_FP16)
            out[m_idx * m_view : m_idx * m_view + m_view, n_idx * n_view : n_idx * n_view + n_view] = out_view


def test_mm_with_l2_split():
    a = torch.randn([M, K], dtype=torch.float16, device="npu:0")
    b = torch.randn([K, N], dtype=torch.float16, device="npu:0")
    out = torch.empty(M, N, dtype=torch.float16, device="npu:0")
    matmul_l2_split_kernel(a, b, out)


if __name__ == "__main__":
    test_mm_with_l2_split()
```
