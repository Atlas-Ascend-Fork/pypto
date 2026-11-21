# pypto.set_convbp_input_tile_shapes

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Sets the TileShape size of each dimension in the L1/L0 buffer hierarchy during conv backward input computation.

The conv backward Dx operator can be considered as matrix multiplication `(HinWin, CoutKhKw) × (CoutKhKw, Cin)`, where the M, N, and K axes correspond to `HinWin`, `Cin`, and `Cout×Kh×Kw`, respectively.

## Prototype

```python
set_convbp_input_tile_shapes(tile_l1_info: pypto_impl.ConvBpTileL1Info, tile_l0_info: pypto_impl.ConvBpTileL0Info) -> None
```

## Parameters

| Parameter  | Input/Output| Description                                        |
|----------|-----------|----------------------------------------------|
| tile_l1_info  | Input     | TileShape configuration information for conv backward computation at the L1 Buffer level, including three parameters: tileML1, tileNL1, and tileKL1. For details, see [Constraints](#constraints).|
| tile_l0_info  | Input     | TileShape configuration information for conv backward computation at the L0 Buffer level, including three parameters: tileML0, tileNL0, and tileKL0. For details, see [Constraints](#constraints).|

## Return Value

void

## Constraints

TileShape must meet the following constraints:

- Alignment constraints:

    - The value of each dimension in ConvBpTileL1Info must satisfy the following constraints:

        - tileML1: Must be less than Win or an integer multiple of Win, which means `tileML1 <= Win` or `tileML1 % Win == 0` (HinWin tiling, ensuring that the M direction does not cross Win rows).

        - tileNL1: Must be a multiple of Cin0, which means `tileNL1 % 16 == 0`.

        - tileKL1: Must be a multiple of `Cout0 × Kh × Kw`, which means `tileKL1 % (16 * Kh * Kw) == 0`.

    - The value of each dimension in ConvBpTileL0Info must satisfy the alignment constraints:

        - tileML0: Must be a multiple of 16, which means `tileML0 % 16 == 0`, and `tileML0 <= CeilAlign(tileML1, 16)`.

        - tileNL0: Must be a multiple of 16, which means `tileNL0 % 16 == 0`, and `tileNL0 <= tileNL1`.

        - tileKL0: Must be a multiple of 16, which means `tileKL0 % 16 == 0`, and `tileKL0 <= tileKL1`.

    Where:

    - `Cin0 = 16`
    - `Cout0 = 16`

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
        (L1Weight + houtL1 * woutL1 * coutL1) * sizeof(dtype) <= L1_size
        ```

        Where:

        - `L1Weight = tileKL1 * tileNL1`
        - `houtL1 = min(CeilDiv(tileML1, Win) + (Kh - 1) * dilationH, (Hout - 1) * strideH + 1)`
        - `woutL1 = min(min(tileML1, Win) + (Kw - 1) * dilationW, (Wout - 1) * strideW + 1)`
        - `coutL1 = tileKL1 / (Kh * Kw)`
        - `L1_size = 524288 bytes` (512KB)
        - `dtype is the data type of the input matrix.`
        - `CeilDiv(a, b) = (a + b - 1) // b`

## Examples

```python
# Construct the L1 tile configuration (ensure that each value is within the valid range).
l1_tile_info = pypto.pypto_impl.ConvBpTileL1Info(
    tileML1=16,    # HinWin tiling, which must satisfy <= Win or % Win == 0.
    tileNL1=16,    # Cin tiling, which must be a multiple of 16.
    tileKL1=144    # Cout*Kh*Kw tiling, which must be a multiple of 16*Kh*Kw (for example, 16*9=144 when Kh=3 and Kw=3).
)

# Construct the L0 tile configuration (satisfying alignment constraints).
l0_tile_info = pypto.pypto_impl.ConvBpTileL0Info(
    tileML0=16,    # The value must be a multiple of 16 and no more than CeilAlign(tileML1, 16).
    tileNL0=16,    # The value must be a multiple of 16 and no more than tileNL1.
    tileKL0=16     # The value must be a multiple of 16 and no more than tileKL1.
)

# Set the TileShape for the conv backward.
pypto.set_convbp_input_tile_shapes(l1_tile_info, l0_tile_info)
```
