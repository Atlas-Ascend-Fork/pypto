# pypto.set_convbp_input_tile_shapes

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:31:26.985Z pushedAt=2026-10-28T11:57:29.594Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->

## Description

Sets the TileShape (slice shape) size of each dimension at the L1/L0 cache level in the convolution backward input computation.

The convolution backward Dx operator can be regarded as matrix multiplication `(HinWin, CoutKhKw) × (CoutKhKw, Cin)`, where the M/N/K axes correspond to `HinWin` / `Cin` / `Cout×Kh×Kw`, respectively.

## Prototype

```python
set_convbp_input_tile_shapes(l1_info: pypto_impl.ConvBpTileL1Info, l0_info: pypto_impl.ConvBpTileL0Info) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| l1_info  | Input      | TileShape configuration information for the convolution backward input computation at the L1 cache level, containing three parameters: tileML1, tileNL1, and tileKL1. See [Constraints](#constraints). |
| l0_info  | Input      | TileShape configuration information for the convolution backward input computation at the L0 cache level, containing three parameters: tileML0, tileNL0, and tileKL0. See [Constraints](#constraints). |

## Return Value

void

## Constraints

TileShape must meet the following constraints:

- Alignment constraints:

    - Each dimension value of ConvBpTileL1Info must satisfy the range constraints:

        - tileML1: Must be less than Win or a multiple of Win, that is, `tileML1 <= Win` or `tileML1 % Win == 0` (HinWin is split along the combined axis, ensuring that the M direction does not cross Win rows).

        - tileNL1: Must be a multiple of Cin0, that is, `tileNL1 % 16 == 0`.

        - tileKL1: Must be a multiple of `Cout0 × Kh × Kw`, that is, `tileKL1 % (16 * Kh * Kw) == 0`.

    - Each dimension of ConvBpTileL0Info must satisfy the alignment constraints:

        - tileML0: Must be a multiple of 16, that is, `tileML0 % 16 == 0`, and `tileML0 <= CeilAlign(tileML1, 16)`.

        - tileNL0: Must be a multiple of 16, that is, `tileNL0 % 16 == 0`, and `tileNL0 <= tileNL1`.

        - tileKL0: Must be a multiple of 16, that is, `tileKL0 % 16 == 0`, and `tileKL0 <= tileKL1`.

    where:

    - `Cin0 = 16`
    - `Cout0 = 16`

- Buffer spatial constraints:

    - L0A, L0B, and L0C spatial constraints:

        ```txt
        tileML0 * tileKL0 * sizeof(dtype) <= L0A_size

        tileKL0 * tileNL0 * sizeof(dtype) <= L0B_size

        tileML0 * tileNL0 * sizeof(FP32) <= L0C_size
        ```

        where:

        - `L0A_size = 65536 bytes`
        - `L0B_size = 65536 bytes`
        - `L0C_size = 131072 bytes`

    - L1 spatial constraints:

        ```txt
        (L1Weight + houtL1 * woutL1 * coutL1) * sizeof(dtype) <= L1_size
        ```

        where:

        - `L1Weight = tileKL1 * tileNL1`
        - `houtL1 = min(CeilDiv(tileML1, Win) + (Kh - 1) * dilationH, (Hout - 1) * strideH + 1)`
        - `woutL1 = min(min(tileML1, Win) + (Kw - 1) * dilationW, (Wout - 1) * strideW + 1)`
        - `coutL1 = tileKL1 / (Kh * Kw)`
        - `L1_size = 524288 bytes` (512KB)
        - `dtype is the data type of the input matrix`
        - `CeilDiv(a, b) = (a + b - 1) // b`

## Examples

```python
# Construct the L1 tile configuration (ensure each value is within the valid range).
l1_tile_info = pypto.pypto_impl.ConvBpTileL1Info(
    tileML1=16,    # HinWin split, must satisfy <= Win or % Win == 0
    tileNL1=16,    # Cin split, must be a multiple of 16.
    tileKL1=144    # Cout*Kh*Kw split, must be a multiple of 16*Kh*Kw (for example, 16*9=144 when Kh=3 and Kw=3).
)

# Construct the L0 tile configuration (satisfying the alignment constraints).
l0_tile_info = pypto.pypto_impl.ConvBpTileL0Info(
    tileML0=16,    # Must be a multiple of 16 and no greater than CeilAlign(tileML1, 16).
    tileNL0=16,    # Must be a multiple of 16 and no greater than tileNL1.
    tileKL0=16     # Must be a multiple of 16 and no greater than tileKL1.
)

# Set the convolution backward input tile shape.
pypto.set_convbp_input_tile_shapes(l1_tile_info, l0_tile_info)
```
