# 约束与公式速查

实现校验链（BpCheck*/FilterCheck*）时的唯一公式来源。所有上限常量：`MAX_SIZE=1000000`、`MAX_PAD_KERNEL=255`、`MAX_DILATION_STRIDE=63`、`SHAPE_INNER_AXIS_MAX_SIZE=65535`、`ALIGN_SIZE_32=32`、`MKN_M_VALUE=MKN_N_VALUE=16`、`Conv::MAX_LOOP=5000`。

## 1. shape 与属性约束（dx/dw 通用）

| 约束项 | 取值范围 |
|--------|---------|
| Batch / Cin / Cout / H / W（各输入输出维度） | [1, 1000000] |
| Kh, Kw | [1, 255] |
| strides | 与卷积维度同长，[1, 63]，单向参数 |
| dilations | 与卷积维度同长，[1, 63]，单向参数 |
| paddings | 2×卷积维度，[0, 255]，双向参数 |
| padding vs kernel | 每轴双侧 padding < (K-1)×dilation+1（dilated kernel） |
| groups | 首版仅支持 1 |
| dtype | DT_FP16 / DT_BF16（out_dtype 与输入一致） |

Load3D 限制：
- dx：`Kh × Kw × 16 ≤ 65535`（16 = Cout0，K 块）
- dw：`Kh × Kw × (32/sizeof(dtype)) ≤ 65535`（Cin0，N 内块；fp16/bf16 下同为 16）
- dx 额外限制：Cout 每组必须是 c0(32B/dtype) 的整数倍（transdata 不消除输出 pad）

一致性校验：
- dx：grad_output.Cout == weight.Cout；input_size 与 grad_output 满足反向卷积关系（见 dx-design §1）
- dw：grad_output.Batch == input.Batch；input.Cin == filterSize.Cin × groups；grad_output.Cout == filterSize.Cout 且整除 groups；Hout/Wout 与 input 满足正向卷积关系（见 dw-design §1）

## 2. tileshape 约束

### dx（M=HiWi，K=CoutHkWk，N=Cin）

| 参数 | 约束 |
|------|------|
| tileML1 | < Win 或 Win 的整数倍（M 不跨 W 行） |
| tileNL1 | 16（Cin0）的倍数 |
| tileKL1 | 16（Cout0）× Kh × Kw 的倍数（HkWk 全载） |
| tileML0 / tileNL0 / tileKL0 | 16 的倍数，且 ≤ CeilAlign(对应 L1 tile, 16) |

### dw（M=Cout，N=CinHkWk，K=HoWo）

| 参数 | 约束 |
|------|------|
| tileML1 | 16 的倍数 |
| tileNL1 | Cin0(32B/dtype) × Kh × Kw 的倍数（HkWk 全载） |
| tileKL1 | < Wout 或 Wout 的整数倍（K 不跨 W 行） |
| tileML0 / tileNL0 / tileKL0 | 16 的倍数，且 ≤ CeilAlign(对应 L1 tile, 16) |

通用：`ConvBpTile::valid()` 要求六元组全正且 L1 ≥ L0。

## 3. L0 / L1 容量公式（A2/A3：L1=512KB，L0A=L0B=64KB，L0C=128KB）

### L0（dx/dw 同构，注意容量取 dtypesize）

```
tileML0 × tileKL0 × sizeof(dtype) ≤ 64KB     // L0A
tileKL0 × tileNL0 × sizeof(dtype) ≤ 64KB     // L0B
tileML0 × tileNL0 × sizeof(float)  ≤ 128KB   // L0C，FP32 累加
```

### dx 的 L1

```
L1Weight = CeilAlign(CeilAlign(tileKL1,16) × tileNL1 × sizeof(dtype), 32)
hinL1  = tileML1 % win == 0 ? tileML1/win : 1        // 反推 L1 上的 fmap 行列
winL1  = tileML1 % win == 0 ? win : tileML1
houtL1 = min(hinL1 + (kh-1)*dilationH, hout)         // 前放大后的 grad_output 区间
woutL1 = min(winL1 + (kw-1)*dilationW, wout)
L1GradOutput = CeilAlign(houtL1 × woutL1 × CeilAlign(tileKL1,16) × sizeof(dtype), 32)
L1Weight + L1GradOutput ≤ 512KB                     // （bias 预留 + tileNL0 对齐量）
```

### dw 的 L1

```
L1Dy = CeilAlign(tileML1 × tileKL1 × sizeof(dtype), 32)          // grad_output 无空间扩展
houtL1 = CeilDiv(tileKL1, wout) ； woutL1 = min(wout, tileKL1)
hinL1  = (houtL1 - 1) × strideH + (kh-1)×dilationH + 1           // fmap 正向扩展
winL1  = (woutL1 - 1) × strideW + (kw-1)×dilationW + 1
L1Fmap = CeilAlign((tileNL1 / (kh×kw)) × hinL1 × winL1 × sizeof(dtype), 32)
L1Dy + L1Fmap ≤ 512KB
```

## 4. 编译耗时门槛（测试用例选择）

```
dx: numTileL0 = batch × groups × CeilDiv(cin, tileNL0) × CeilDiv(hin×win, tileML0)
    K = CeilAlign(cout,16) × kh × kw
    tileTotalNum ≈ numTileL0 × CeilDiv(K, tileKL0)
dw: numTileL0 = batch × groups × CeilDiv(cout, tileML0) × CeilDiv(cin×kh×kw, tileNL0)
    tileTotalNum ≈ numTileL0 × CeilDiv(hout×wout, tileKL0)       // batch 已计入 numTileL0
```

- 门槛用例 tileTotalNum < 1000，最大不超过 2000，否则编译耗时明显偏高。
- tileTotalNum > 5000（MAX_LOOP）时打印警告日志：「Suggestion: Consider increasing tile size to reduce compilation time.」

## 5. 区间反算核心公式（TileGraph 用）

### dx：M tile → grad_output 区间（前放大方向）

```
hinL1Offset = mL1Offset / win ； winL1Offset = mL1Offset % win
houtL1Offset = hinL1Offset - bpPadUp          // bpPadUp = (kh-1)*dilationH - padTop，可为负
endHout = houtL1Offset + hinL1Size + (kh-1)*dilationH
houtL1Size = clamp(endHout, 0, expandHout) - max(houtL1Offset, 0)   // 与 [0, expandHout) 交集
expandHout = (hout-1)*strideH + 1 ； expandWout = (wout-1)*strideW + 1
```

### dw：K tile → fmap 区间（正向扩展方向）

```
houtL1Offset = kL1Offset / wout ； woutL1Offset = kL1Offset % wout
hinL1Offset = houtL1Offset × strideH - padTop    // 可为负（pad 区）
endHin = hinL1Offset + (houtL1Size-1)×strideH + (kh-1)×dilationH + 1
hinL1Size = clamp(endHin, 0, hin) - max(hinL1Offset, 0)    // 与 [0, hin) 交集；L1 只搬有效区
```

## 6. 参考用例 shape 与 tile 配置

dx（需求文档 demo）：N=1, Cin=16, Cout=16, Hin=5, Win=32, Kh=Kw=3, stride=1, pad=1, dil=1, FP16；tile L1 (M=16, K=144, N=16)、L0 (16,16,16)。

dw（需求文档 demo）：N=1, Cin=16, Cout=16, Hin=4, Win=15, Kh=Kw=2, stride=1, pad=1, dil=1, FP16；Hout=5, Wout=16；tile L1 (M=16, K=80, N=64)、L0 (M=16, K=80, N=64)。

两个 demo 均满足上述全部约束公式，可作为校验链的冒烟用例。
