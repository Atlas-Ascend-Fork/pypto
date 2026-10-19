# TileOp（pto-isa）编写模式

TileOp 是运行在 Cube 核上的 C++ 模板函数，由 codegen 生成为 tile 级语句调用。文件位于 `framework/src/interface/tileop/conv/conv_bp_pto.h`，统一被 `TileOpImpl.h` include。

## 1. 函数签名约定

```cpp
template <bool isConv3D, typename T, typename U>
TILEOP void TLoadConvBpDxDedy(T& dst, U& src, const int64_t& offset0, ...);

template <typename T, typename U>
TILEOP void TLoad2DDX(T& dst, U& src, const int64_t& kL0Size, ...);

template <CopyOutMode mode, bool isConv3D, typename T, typename U>
TILEOP void TStoreConvDW(T& dst, U& src, ...);
```

- `T` = 目的（L1/L0/GM tile 视图），`U` = 源；修饰符 TILEOP（tileop 主体）/ INLINE（辅助函数）/ `__tf__ INLINE`（fixpipe 辅助）。
- 参数全部为值/const 引用，由 codegen 按 opcode 注册的 attr 白名单 + tileOpParams 顺序拼出。
- 参数顺序必须与 codegen Gen 函数生成的实参顺序严格一致——这是 opcode 注册、TileGraph SetAttribute、codegen、TileOp 签名四处对齐的结果。

## 2. 静态/动态 shape 读取

```cpp
// 编译期静态 shape（tile 模板参数）
constexpr auto stcDstH = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
// 运行期实际 shape / stride（不满块 tile）
int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
int64_t srcStride2 = GetConvStride<CONV_IDX_2>(src);
```

- `CONV_IDX_0..4` 为 tile 视图的轴下标；`BLOCK_ALIGN_BYTE / sizeof(Type)` 得 c0Size（32B 对齐块）。
- L1 tile 的 shape 用 `dstAL1TensorPtr->UpdateDynValidShape(...)` 与静态 shape 保持一致，供 L1 复用 pass 判断。

## 3. 常用类型

| 类型 | 用途 |
|------|------|
| `pto::Tile<TileType::Mat, Type, ROW, COL, BLayout::ColMajor, -1, -1, SLayout::RowMajor>` | 通用分型 tile（L1 内连续块） |
| `pto::TileLeft<Type, M, K, -1, -1>` | L0A 目的（ZZ 布局） |
| `pto::TileRight<Type, M, N, -1, -1>` | L0B 目的（NZ 布局） |
| `pto::ConvTile<TileType::Mat, Type, BUFFER_BYTES, pto::Layout::NDC1HWC0, pto::ConvTileShape<...>>` | 卷积 5HD 搬运单位（buffer 尺寸需 2 的幂对齐） |
| `pto::GlobalTensor<Type, Shape<...>, Stride<...>, pto::Layout::NDC1HWC0/NZ>` | GM 视图（支持 GM 侧 stride） |

## 4. 原语清单

| 原语 | 语义 |
|------|------|
| `pto::TASSIGN(tile, addr)` | 把 tile 绑定到地址（`dst.GetAddr() + offset*sizeof(Type)`） |
| `pto::TLOAD(dstTile, srcGlobal)` | GM→L1 搬运（走 MTE2） |
| `pto::TEXTRACT<DstType, SrcType>(dst, src, posM, posK)` | L1→L0 转置搬运（走 MTE1），pos 定位 L0 内子块 |
| `pto::TFILLPAD(zeroTile, zeroTile)` | tile 清零（写 0 填充） |
| `pto::TSTORE` / fixpipe | L0C→GM 搬出 |
| `pipe_barrier(PIPE_MTE2)` | 搬入后同步（清零后、下次写前） |
| `set_fmatrix_b` / `set_l3d_rpt` / `set_padding` | img2colv2 前置寄存器 |
| `img2colv2_cbuf_to_cb(...)` | L1(cbuf)→L0B(cb) img2col 硬件指令 |

## 5. 三个核心套路

### 套路 A：清零 + 逐块搬运（带间隔/稀疏的 GM→L1）

pto-isa 不支持 L1 侧 stride，L1 数据必须连续。因此凡是目标 L1 布局与 GM 布局存在间隔（dx 前放大）或分型（dw 的 HoWo/16）差异的，都用：

1. zeroTile 覆盖整个 L1 tile 区域 TFILLPAD 清零（覆盖尾部不满块/pad 区）；
2. `pipe_barrier`；
3. 外层循环按最小分型（通常一个 c0 块或一个 ConvTile）逐块 TLOAD，目的偏移在循环内手工累加，源偏移按 GM stride 累加。

### 套路 B：TEXTRACT 转置（L1→L0）

- 需要「外布局不动、内分型转置」时（N0C0→C0N0 等）：构造小的源/目的 tile 类型对，一次 `TEXTRACT<Dst, Src>(dst, src, posM, posK)` 搬一个 [16, c0] 块，循环控制块间偏移。
- 需要逆序（dx 的 HW 逆序）时：在循环里对块索引做 `hwk - idx - 1` 映射后再算源偏移。
- posM/posK 是 L0 tile 在 L1 大块中的偏移（以元素计，uint16 截断注意范围）。

### 套路 C：寄存器配置 + 硬件指令（img2col）

见 dw-design.md §6 的 TImg2ColToL0B：三个 set_xxx 寄存器（fmatrix_b/l3d_rpt/padding）逐 bit 组装后，下发 `img2colv2_cbuf_to_cb`；注意指令参数顺序（stepK, stepM, posK, posM）与接口直觉相反，filter > 255 需拆 low/high。

## 6. 编写检查项

- [ ] 模板参数与修饰符和同文件既有 tileop 一致
- [ ] 静态 shape 从 TileShape 元组取，动态 shape 从 GetConvShape 取，二者不混用
- [ ] c0Size 一律 `BLOCK_ALIGN_BYTE / sizeof(Type)` 推导，不硬编码
- [ ] 目的偏移乘 `sizeof(Type)` 换算字节
- [ ] 清零后必须 pipe_barrier 才能开始搬入
- [ ] 2D/3D 统一：isConv3D=false 时 D 轴 stride 置 0、shape 置 1，寻址统一按 5 轴
- [ ] 函数名与 opcode.cpp 注册的 `TileOp::` 前缀名一致
