# pypto.set_convbp_filter_tile_shapes

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

Sets the TileShape size of each dimension in the L1/L0 buffer hierarchy during conv backward filter computation.

The conv backward filter gradient Dw operator can be considered as matrix multiplication `(Cout, HoutWout) × (HoutWout, CinKhKw)`, where the M, N, and K axes correspond to `Cout`, `Cin × Kh × Kw`, and `Hout × Wout` (cross-batch accumulation), respectively.

## Prototype

```python
set_convbp_filter_tile_shapes(tile_l1_info: pypto_impl.ConvBpTileL1Info, tile_l0_info: pypto_impl.ConvBpTileL0Info) -> None
```

## Parameters

| Parameter  | Input/Output| Description                                        |
|----------|-----------|----------------------------------------------|
| tile_l1_info  | Input     | TileShape configuration information for conv backward filter computation at the L1 Buffer level, including three parameters: tileML1, tileNL1, and tileKL1. For details, see [Constraints](#constraints).|
| tile_l0_info  | Input     | TileShape configuration information for conv backward filter computation at the L0 Buffer level, including three parameters: tileML0, tileNL0, and tileKL0. For details, see [Constraints](#constraints).|

## Return Value

void

## Constraints

TileShape must meet the following constraints:

- Alignment constraints:

    - The value of each dimension in ConvBpTileL1Info must satisfy the following constraints:

        - tileML1: Must be a multiple of 16, which means `tileML1 % 16 == 0` (Cout tiling).

        - tileNL1: Must be a multiple of `Cin0 × Kh × Kw`, which means `tileNL1 % (16 * Kh * Kw) == 0` (CinKhKw tiling, ensuring Cin0×Kh×Kw alignment in the N direction).

        - tileKL1: Must be less than Wout or an integer multiple of Wout, which means `tileKL1 <= Wout` or `tileKL1 % Wout == 0` (HoutWout tiling along the K axis, ensuring that the im2col expansion does not cross Wout rows.)

    - The value of each dimension in ConvBpTileL0Info must satisfy the alignment constraints:

        - tileML0: Must be a multiple of 16, which means `tileML0 % 16 == 0`, and `tileML0 <= CeilAlign(tileML1, 16)`.

        - tileNL0: Must be a multiple of 16, which means `tileNL0 % 16 == 0`, and `tileNL0 <= CeilAlign(tileNL1, 16)`.

        - tileKL0: Must be a multiple of 16, which means `tileKL0 % 16 == 0`, and `tileKL0 <= CeilAlign(tileKL1, 16)`.

    Where:

    - `Cin0 = 16`
    - `Cout0 = 16`
    - `CeilAlign(a, b)` indicates that a is rounded up to a multiple of b.

- Buffer space constraints:

    - L0A Buffer, L0B Buffer, and L0C Buffer space constraints:

        ```txt
        tileML0 * tileKL0 * sizeof(dtype) <= L0A_size

        tileKL0 * tileNL0 * sizeof(dtype) <= L0B_size

        tileML0 * tileNL0 * sizeof(FP32) <= L0C_size
        ```

        Where:

        - `L0A_size = 65536 bytes`
        - `L0B_size = 65536 bytes`
        - `L0C_size = 131072 bytes`

    - L1 Buffer space constraints:

        ```txt
        (gradOutputL1Size + inputL1Size) <= L1_size
        ```

        Where:

        - `gradOutputL1Size = CeilAlign(tileML1 * tileKL1 * sizeof(dtype), 32)`
        - `houtL1 = CeilDiv(tileKL1, Wout)`, `woutL1 = min(Wout, tileKL1)`
        - `hinL1 = (houtL1 - 1) * strideH + (Kh - 1) * dilationH + 1`
        - `winL1 = (woutL1 - 1) * strideW + (Kw - 1) * dilationW + 1`
        - `inputL1Size = CeilAlign((tileNL1 / (Kh * Kw)) * hinL1 * winL1 * sizeof(dtype), 32)`
        - `L1_size = 524288 bytes`
        - `dtype is the data type of the input matrix.`
        - `CeilDiv(a, b) = (a + b - 1) // b`

## Examples

```python
# Construct the L1 tile configuration (ensure that each value is within the valid range).
l1_tile_info = pypto.pypto_impl.ConvBpTileL1Info(
    tileML1=32,     # Cout tiling, which must be a multiple of 16.
    tileKL1=32,     # HoutWout tiling, which must satisfy <= Wout or % Wout == 0.
    tileNL1=144     # Cin*Kh*Kw tiling, which must be a multiple of 16*Kh*Kw (for example, 16*9=144 when Kh=3 and Kw=3).
)

# Construct the L0 tile configuration (satisfying alignment constraints).
l0_tile_info = pypto.pypto_impl.ConvBpTileL0Info(
    tileML0=16,    # The value must be a multiple of 16 and no more than CeilAlign(tileML1, 16).
    tileNL0=16,    # The value must be a multiple of 16 and no more than CeilAlign(tileNL1, 16).
    tileKL0=16     # The value must be a multiple of 16 and no more than CeilAlign(tileKL1, 16).
)

# Set the TileShape for the conv backward filter gradient.
pypto.set_convbp_filter_tile_shapes(l1_tile_info, l0_tile_info)
```
