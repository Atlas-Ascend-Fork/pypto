# pypto.set_convbp_filter_tile_shapes

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

设置卷积反向权重梯度（conv backward filter）计算中L1/L0缓存层级下各维度的TileShape（切片形状）大小。

卷积反向权重梯度Dw算子可视为矩阵乘法 `(Cout, HoutWout) × (HoutWout, CinKhKw)`，其中M/N/K三轴分别对应 `Cout` / `Cin × Kh × Kw` / `Hout × Wout`（跨Batch累加）。

## 函数原型

```python
set_convbp_filter_tile_shapes(tile_l1_info: pypto_impl.ConvBpTileL1Info, tile_l0_info: pypto_impl.ConvBpTileL0Info) -> None
```

## 参数说明

| 参数名   | 输入/输出 | 说明                                         |
|----------|-----------|----------------------------------------------|
| tile_l1_info  | 输入      | L1缓存层级下卷积反向权重梯度计算的TileShape配置信息，有tileML1、tileNL1、tileKL1三个参数。请参考[约束说明](#约束说明)章节填写 |
| tile_l0_info  | 输入      | L0缓存层级下卷积反向权重梯度计算的TileShape配置信息，有tileML0、tileNL0、tileKL0三个参数。请参考[约束说明](#约束说明)章节填写 |

## 返回值说明

void

## 约束说明

TileShape需要满足以下约束条件：

- 对齐约束：

    - ConvBpTileL1Info各维度值需满足范围约束：

        - tileML1：需为16的倍数，即 `tileML1 % 16 == 0`（Cout切分）

        - tileNL1：需为 `Cin0 × Kh × Kw` 的倍数，即 `tileNL1 % (16 * Kh * Kw) == 0`（CinKhKw合轴切分，N方向保证Cin0×Kh×Kw对齐）

        - tileKL1：需小于Wout或者为Wout的整数倍，即 `tileKL1 <= Wout` 或 `tileKL1 % Wout == 0`（K轴为HoutWout合轴切分，保证im2col展开不跨Wout行）

    - ConvBpTileL0Info各维度值需满足对齐约束：

        - tileML0：需为16的倍数，即 `tileML0 % 16 == 0`，且 `tileML0 <= CeilAlign(tileML1, 16)`

        - tileNL0：需为16的倍数，即 `tileNL0 % 16 == 0`，且 `tileNL0 <= CeilAlign(tileNL1, 16)`

        - tileKL0：需为16的倍数，即 `tileKL0 % 16 == 0`，且 `tileKL0 <= CeilAlign(tileKL1, 16)`

    其中：

    - `Cin0 = 16`
    - `Cout0 = 16`
    - `CeilAlign(a, b)` 表示将a向上对齐到b的倍数

- buffer空间约束：

    - L0A Buffer、L0B Buffer、L0C Buffer空间约束：

        ```txt
        tileML0 * tileKL0 * sizeof(dtype) <= L0A_size

        tileKL0 * tileNL0 * sizeof(dtype) <= L0B_size

        tileML0 * tileNL0 * sizeof(FP32) <= L0C_size
        ```

        其中：

        - `L0A_size = 65536 bytes`
        - `L0B_size = 65536 bytes`
        - `L0C_size = 131072 bytes`

    - L1 Buffer空间约束：

        ```txt
        (gradOutputL1Size + inputL1Size) <= L1_size
        ```

        其中：

        - `gradOutputL1Size = CeilAlign(tileML1 * tileKL1 * sizeof(dtype), 32)`
        - `houtL1 = CeilDiv(tileKL1, Wout)`，`woutL1 = min(Wout, tileKL1)`
        - `hinL1 = (houtL1 - 1) * strideH + (Kh - 1) * dilationH + 1`
        - `winL1 = (woutL1 - 1) * strideW + (Kw - 1) * dilationW + 1`
        - `inputL1Size = CeilAlign((tileNL1 / (Kh * Kw)) * hinL1 * winL1 * sizeof(dtype), 32)`
        - `L1_size = 524288 bytes`
        - `dtype为输入矩阵的数据类型`
        - `CeilDiv(a, b) = (a + b - 1) // b`

## 调用示例

```python
# 构造L1 Tile配置（确保各值在合法范围）
l1_tile_info = pypto.pypto_impl.ConvBpTileL1Info(
    tileML1=32,     # Cout切分，需为16的倍数
    tileKL1=32,     # HoutWout切分，需满足 <= Wout 或 % Wout == 0
    tileNL1=144     # Cin*Kh*Kw切分，需为16*Kh*Kw的倍数（如Kh=3,Kw=3时16*9=144）
)

# 构造L0 Tile配置（满足对齐约束）
l0_tile_info = pypto.pypto_impl.ConvBpTileL0Info(
    tileML0=16,    # 需为16的倍数，且不大于CeilAlign(tileML1, 16)
    tileNL0=16,    # 需为16的倍数，且不大于CeilAlign(tileNL1, 16)
    tileKL0=16     # 需为16的倍数，且不大于CeilAlign(tileKL1, 16)
)

# 设置卷积反向权重梯度TileShape
pypto.set_convbp_filter_tile_shapes(l1_tile_info, l0_tile_info)
```
