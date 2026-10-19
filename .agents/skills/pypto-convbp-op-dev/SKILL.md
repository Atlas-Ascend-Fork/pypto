---
name: pypto-convbp-op-dev
description: 在 A2/A3 (DAV_2201) 平台上开发 PyPTO 卷积反向算子 dx（conv_backward_input 输入梯度）与 dw（conv_backward_filter 权重梯度），覆盖 TileShape 体系、opcode 注册、TileOp pto-isa 实现、TensorGraph/TileGraph 循环展开、NPU codegen、Python API 与 bindings、UT/ST 测试及文档的全栈开发流程。当用户要求开发、实现或生成卷积反向算子、conv backward 算子、convbp、dx 算子、dw 算子、输入梯度算子、权重梯度算子，或要求在 pto 框架上支持 conv_backward_input / conv_backward_filter 接口时，必须使用本 skill。
---

# PyPTO 卷积反向算子（dx/dw）开发指南

在 PyPTO 框架上开发 A2/A3 平台的卷积反向算子。dx 与 dw 共享同一套分层框架（TileShape → Opcode → TileOp → Operation → Codegen → Python），但矩阵乘分解、循环层级与 TileOp 集合不同。

## 输入

用户会提供：
- **算子类型**：`dx`（输入梯度）、`dw`（权重梯度）或两者
- 可选：shape/dtype/stride/pad/dilation 范围、性能要求

示例输入：
> 请在 pypto 上实现 A2/A3 的卷积反向 dw 算子（conv_backward_filter）。

## 重要约束

- 本 skill 提供设计规范、命名约定、公式与实现模式，**不提供完整实现代码**。所有函数体必须依据各阶段的设计要点自行编写。
- 仅支持 **A2/A3（NPUArch::DAV_2201）**，入口函数必须用 `CHECK(FwkErr::PLATFORM_NOT_SUPPORTED, ...)` 校验平台。
- 首版功能边界：不支持量化/bias/分组卷积(groups>1)/FP32 计算/1D/3D 首版落地（dx 框架预留 1D/3D 分支）/A5/动态 shape/output_paddings/空 Tensor。dx/dw 均支持 FP16/BF16 输入输出。
- 命名必须与 [references/file-change-map.md](references/file-change-map.md) 中的清单一致（opcode 名、TileOp 名、Python API 名、配置 key 名），这是多模块协同的前置条件。

## 参考文件

| File | Purpose | Load Timing |
|------|---------|-------------|
| [references/convbp-architecture.md](references/convbp-architecture.md) | 卷积反向的矩阵乘分解、内存层级数据流、格式转换、dx/dw 差异对照表 | 阶段 1 开始时读取 |
| [references/file-change-map.md](references/file-change-map.md) | dx/dw 全量文件变更地图：每个文件的修改要点、命名清单、注册链路 | 阶段 1 开始时读取，后续各阶段对照使用 |
| [references/dx-design.md](references/dx-design.md) | dx 专属设计：循环结构、L1 有效区间反算、grad_output 前放大、weight 转置逆序 | 开发 dx 时（阶段 1、5、6）读取 |
| [references/dw-design.md](references/dw-design.md) | dw 专属设计：K 轴跨 batch 累加、fmap 有效区间搬入、FZ 输出、4 个新 TileOp | 开发 dw 时（阶段 1、5、6）读取 |
| [references/tileop-patterns.md](references/tileop-patterns.md) | pto-isa TileOp 编写模式：类型系统、原语清单、常用套路 | 阶段 4 开始时读取 |
| [references/constraints-and-formulas.md](references/constraints-and-formulas.md) | shape/属性/tileshape 约束、L1/L0 容量公式、编译耗时门槛公式 | 阶段 2、5 编写校验时读取 |
| [references/testing-and-golden.md](references/testing-and-golden.md) | golden 对拍（aten.convolution_backward）、UT/ST 组织、精度阈值 | 阶段 8 开始时读取 |
| [scripts/check_convbp_status.py](scripts/check_convbp_status.py) | 确定性检查 pypto 仓中 dx/dw 各模块是否已存在（增量开发判断） | 阶段 0 执行、阶段 9 复验 |

## 前置条件

- pypto 源码仓可访问。本 skill 位于 `<pypto-root>/.agents/skills/pypto-convbp-op-dev/`，仓根即本 skill 目录向上 3 级；若不确定，用 `git rev-parse --show-toplevel` 在 skill 目录下确认。
- 正向 conv 算子（`Conv` 命名空间、`OP_L1_COPY_IN_CONV`/`OP_LOAD3D_CONV`/`OP_L0C_COPY_OUT_CONV`/`TILE_A_MUL_B`）已存在，dx/dw 大量复用正向 TileOp。
- 如需运行测试，需要已编译的 pypto 环境。

## 工作流程

按 10 个阶段（0~9）顺序执行。阶段 2~7 有严格依赖关系（底层先行），不得调换顺序。dx 与 dw 同时开发时，先完成 dx 全部阶段再进入 dw（dw 依赖 dx 建立的 `ConvBp` 命名空间与 TileShape 拆分）。

### 阶段 0：现状盘点

1. 定位 pypto 仓根目录（见前置条件）。
2. 运行现状检查脚本：

   ```bash
   python3 <skill-dir>/scripts/check_convbp_status.py --pypto-root <pypto-root> --op dx
   python3 <skill-dir>/scripts/check_convbp_status.py --pypto-root <pypto-root> --op dw
   ```

3. 解读 JSON 输出：
   - 目标算子的全部模块 `missing` → 全新开发，从阶段 1 开始。
   - 部分模块 `present` → 增量开发，仅对 `missing` 模块按阶段补齐。已存在模块原则上只读；唯一例外：开发 dw 时 file-change-map「C1. TileShape 拆分」明确列出的共享模块改造（TileShape/配置 key 拆分、bindings 改名、dx 实现文件改名）属于 dw 变更本身，必须执行。
   - 脚本执行失败 → 报告错误并人工用 `grep -r "OP_CONV_BP_INPUT_2D" framework/` 等方式确认现状，不得跳过盘点直接开发。

### 阶段 1：设计理解

1. 读取 [references/convbp-architecture.md](references/convbp-architecture.md)，明确目标算子的 MKN 轴映射、循环层级、格式转换链路。
2. 读取 [references/file-change-map.md](references/file-change-map.md)，列出本次需要修改/新建的全部文件。
3. 读取对应设计文档（dx 读 [references/dx-design.md](references/dx-design.md)，dw 读 [references/dw-design.md](references/dw-design.md)）。
4. 产出一份模块清单（文件 + 修改要点 + 新建 TileOp/opcode 列表），向用户确认后再动手。这一步防止命名漂移——后续 5 个模块（opcode 注册、TileGraph、TileOp、codegen、bindings）必须引用同一套名字。

**验收**：模块清单覆盖 file-change-map 中目标算子的全部行项，且每个新 opcode/TileOp/API 名与清单一致。

### 阶段 2：TileShape 体系

1. 按 file-change-map「TileShape 模块」小节修改 `framework/include/tilefwk/tilefwk_op.h`、`framework/include/tilefwk/tile_shape.h`、`framework/src/interface/operation/tile_shape.cpp`、`framework/src/interface/configs/config_manager_ng.{h,cpp}`。
2. 核心动作：
   - 在 `npu::tile_fwk::ConvBp` 命名空间定义 `ConvBpTileL1Info`/`ConvBpTileL0Info`（字段 tileML1/tileKL1/tileNL1、tileML0/tileKL0/tileNL0）。
   - `TileShape` 持有 ConvBp tile 成员（dx 首次引入 `ConvBpTile`；引入 dw 时拆分为 `convBpInputTile`/`convBpFilterTile` 两套，配置 key 同步拆分为 `convbp_input_tile_shapes`/`convbp_filter_tile_shapes`）。
   - `ConvBpTile::valid()` 校验 L1/L0 tile 正性与 L1 ≥ L0。
   - `ConfigScope` 的 `HasConfig`/`GenerateTileShape`/`GetConvBp*Tile` 同步登记。
3. M/K/N 语义随算子不同：dx 的 M=HiWi、K=Cout×Kh×Kw、N=Cin；dw 的 M=Cout、N=Cin×Kh×Kw、K=Hout×Wout。切分约束见 [references/constraints-and-formulas.md](references/constraints-and-formulas.md)。

**验收**：全量编译通过；`TileShape::Current().SetConvBp*Tile(...)` 后 `ToString(TileType::CONV_BP)` 输出六元组。

### 阶段 3：Opcode 注册链路

1. 按 file-change-map「Opcode 注册模块」小节，在以下文件注册算子级 opcode 与 TileOp 级 opcode：
   - `framework/src/interface/operation/opcode.h`：enum 增加 opcode（算子级：`OP_CONV_BP_INPUT_2D/3D`（dx）、`OP_CONV_BP_FILTER_2D`（dw）；TileOp 级见 file-change-map）。
   - `framework/src/interface/operation/opcode.cpp`：`RegisterCube()` 中为每个 opcode 调用 `RegisterInfo`，内容包括：OpCoreType::AIC、ISA 字符串名、输入/输出 MemoryType、TileOp 实现名（`TileOp::TLoadConvBpDxDedy` 等）、PIPE 类型、OpCalcType、attr 白名单、`DAV_2201` 支持的 TileOpFormat 集合（输入格式集 → 输出格式集）。同时把新 opcode 加进 `SUPPORT_TILETENSOR_OPS` 与 `IsCopyIn`/`IsLoad2D`/L1 搬入白名单等 opcode 集合判断函数。
   - `framework/src/interface/operation/op_infer_shape_impl.cpp`：为每个新 L1 搬入/Load 算子 `REGISTER_INFER_SHAPE_FUNC`（L1 搬入用基于 `l1_tile_shape` 属性的 InferFunc，Load 类复用 `L1ToL0ConvInferFunc`，L0C 搬出复用 `L0CCopyOutConvInferFunc`）。
   - `framework/src/codegen/codegen_op.cpp`：L1 搬入类 opcode 加入 `OP_SHAPE_FROM_ATTR`。
   - `framework/src/cost_model/simulation/common/ISA.h`：为 CUBE/搬运 ISA 名补 `SCHED_CORE_PIPE_TYPE` 管道映射。
2. `framework/src/interface/operation/operation.{h,cpp}`：新增算子用到的 `OpAttributeKey`（dx：skipH/skipW/hwk/kL0Size/nL0Size/k0Idx/n0Idx 等；dw 复用正向 conv 的 filter/stride/dilation/padding/postM/postK 系列）。

**验收**：编译通过；`grep` 确认每个新 opcode 在 opcode.h/opcode.cpp/SUPPORT_TILETENSOR_OPS 三处成对出现。

### 阶段 4：TileOp 实现（pto-isa）

1. 读取 [references/tileop-patterns.md](references/tileop-patterns.md)，掌握 `pto::Tile`/`pto::TileLeft`/`pto::TileRight`/`pto::ConvTile`/`pto::GlobalTensor` 类型与 TASSIGN/TLOAD/TEXTRACT/TFILLPAD/img2colv2 等原语。
2. 新建 `framework/src/interface/tileop/conv/conv_bp_pto.h`（dx 首建，dw 追加），实现阶段 1 清单中的 TileOp 模板函数：
   - dx：`TLoadConvBpDxDedy`（GM→L1 前放大）、`TLoadConvBPNZ`（GM→L1 FZ 直搬）、`TLoad2DDX`（L1→L0B 转置逆序）。
   - dw：`TLoadConvBPDwDedy`（GM→L1 清零+逐 Cout1 搬入）、`TLoadDwA`（L1→L0A NZ→ZZ 转置）、`TLoad3DDW`（L1→L0B img2col，内含 `TImg2ColToL0B` 辅助函数下发 `img2colv2_cbuf_to_cb`）、`TStoreConvDW`（L0C→GM FZ fixpipe）。
   - 每个 TileOp 的算法意图、数据格式、参数列表见 dx-design/dw-design 对应小节——按设计意图自行实现，不照抄任何既有代码。
3. 在 `framework/src/interface/tileop/TileOpImpl.h` include 该头文件。
4. 实现约束：
   - 模板签名沿用 `template <bool isConv3D, typename T, typename U>` / `template <typename T, typename U>`，修饰符 TILEOP/INLINE 与正向 conv tileop 一致。
   - pto-isa 只支持 GM 侧 stride、L1 内数据必须连续排布，因此跨块/带间隔的搬运一律用「外层循环 + 每次搬一个最小分型」的最简实现。
   - L1 目标区域若存在尾部不满块（HoWo 不满 16 对齐），先 TFILLPAD 清零再搬入。

**验收**：头文件被 TileOpImpl.h 引入后全量编译通过；TileOp 函数名与阶段 3 注册的 `TileOp::` 名一致。

### 阶段 5：Operation 实现（校验 + TensorGraph + TileGraph）

新建 `framework/src/interface/operation/convbp_dx_operation_impl.cpp` 或 `convbp_dw_operation_impl.cpp`（命名空间 `npu::tile_fwk::ConvBp`），并在 `operation_impl.h` 声明共享结构、在 `operation_impl.cpp` 的 `ExpandOperationInto` 增加 case 分发。内部按四段组织：

1. **入口与校验**：`ConvBackwardInput` / `ConvBackwardFilter` 入口（平台校验、维度归一化为 3D 语义、构造 `ConvBpAttrParam`），随后依次调用：OriginShape 范围校验 → OutputShape 关系校验 → AttrShape（groups/Cout/pad<dilated kernel/Load3D 上限）校验 → TileTiling（tile 对齐与范围、L0A/L0B/L0C 容量）校验 → L1SizeTiling 容量校验。全部公式见 [references/constraints-and-formulas.md](references/constraints-and-formulas.md)，超限用 `CHECK`/`ASSERT` 报 `ExternalError`/`ConvOperationError`。
2. **TensorGraph**：构造输出 Tensor（dx 为 NC1HWC0 5HD、dw 为 FRACTAL_Z），插入 `OP_FAKE_TRANS` 完成 ND→NZ/NZ→ND（dw 的 FZ→ND 需同步扩展 `infer_tensor_format.cpp`），并把原始 shape/卷积属性挂在算子节点属性上供 TileGraph 使用。
3. **TileGraph**：按 dx-design/dw-design 的循环结构实现 `ConstructTileGraph` 及各级 `Iter*ExpandFunc`。要点：
   - 迭代状态集中在 `ConvBpIterInfo`，tile/shape 常量集中在 `ConvBpTileInfo`，图形节点句柄集中在 `ConvBpGraphNodes`（结构体定义放 operation_impl.h 供两算子共享）。
   - L1 循环内先 `Update*IterInfo` 计算本 tile 的有效区间（区间反算公式是 dx/dw 各自设计文档的核心，必须逐条实现），无效区间跳过。
   - K 累加用 `isFirstK`/`isLastK` 选择 `TILE_A_MUL_B`（首 K，无累加输入）或 `TILE_A_MULACC_B`（FP32 累加到 `cL0PartialSum`）；dx 的 isLastK 以单 batch 内 K 轴为界，dw 以跨全部 batch 的 K 轴为界——这是两算子循环层级唯一本质差异。
   - 每个 TileOp 节点用 `SetAttribute` 下发阶段 2/3 定义的属性（`l1_tile_shape`/`l0_tile_shape`/`srcGmConvValidShape`/postM/postK/padding/repeat 系列等），并用 `CopyOpAttribute`+`SetOpAttribute` 描述 GM 视图。
4. **构建 pybind**：见阶段 7。

**验收**：`framework/tests/ut/operator`（或 `framework/tests/ut/interface/src/operation/test_operation_impl_conv.cpp`）中新增的算子展开 UT 通过（用例模式见 testing-and-golden.md）。

### 阶段 6：NPU Codegen

1. 新建 `framework/src/codegen/npu/codegen_mte_convbp.cpp`，为每个新 TileOp opcode 实现 `CodeGenOpNPU::GenMem*` 方法：从 `GetOpAttr` 取属性、`GetTileOpParamsByOrder`/`GetOffsetFromAttr` 取参数，拼出 tileOp 调用语句（`tileOpName << WrapParamByParentheses(paramList) << STMT_END`）。
2. 在 `codegen_op_npu.h` 声明、`codegen_op_npu.cpp` 构造函数的 opcode→Gen 映射表登记。可复用已有 Gen（如 dw 的 `OP_LOAD3D_CONV_DW` 复用 `GenMemL1ToL0Load3D`）。
3. 维度断言：L1 搬入的 offset/shape 维数与 isConv3D 匹配（2D 补 D 轴 0/1）。

**验收**：`framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_mte/test_codegen_dyn_convbp.cpp` 通过。

### 阶段 7：Python API 与 bindings

1. `python/pypto/op/convbp.py`：实现 `conv_backward_input`（dx）/`conv_backward_filter`（dw）公开函数（`@op_wrapper` 装饰），私有 `__validate_*` 做类型/维度/dtype 一致性校验，非法输入抛 `PyptoError`；转调 `pypto_impl.ConvBackwardInput`/`ConvBackwardFilter`。在 `python/pypto/op/__init__.py` 导出。
2. `python/pypto/_controller.py`：实现 `set/get_convbp_input_tile_shapes`（dx）与 `set/get_convbp_filter_tile_shapes`（dw），经 `ConvBpTile` 包装写入对应 scope key。`python/pypto/config.py`：`ConvBpTile` 类、`_Options`/`options`/`set_options`/`ConfigScope` 同步。
3. `python/src/bindings/controller.cpp`：绑定 `ConvBpTileL1Info`/`ConvBpTileL0Info` py class 与 `Set/GetConvBpInputTile`、`Set/GetConvBpFilterTile`；`python/src/bindings/operation.cpp`：绑定 `ConvBackwardInput`/`ConvBackwardFilter`。
4. 更新 `python/pypto/pypto_impl/__init__.pyi` stub。

**验收**：`python -c "import pypto; pypto.conv_backward_input; pypto.set_convbp_input_tile_shapes"` 无 AttributeError（注意 .so 双树陷阱，必要时重新 build）。

### 阶段 8：测试

按 [references/testing-and-golden.md](references/testing-and-golden.md) 完成：

1. Python UT（`python/tests/ut/operation/test_convbp_op.py`）：合法用例 + 各类非法输入报错用例。
2. Python ST（`python/tests/st/test_convbp.py`）：NPU/SIM 双模式，golden 用 `torch.ops.aten.convolution_backward`（dx 取 `output_mask=[True, False, False]` 只算输入梯度，dw 取 `[False, True, False]` 只算权重梯度），`allclose(atol=1e-2, rtol=1e-2)`。
3. 框架侧 UT（codegen/operation impl）已在阶段 5/6 就位，此处统一运行。
4. 用例 shape 选择满足 tileTotalNum 门槛（<1000，见 constraints-and-formulas.md），防止编译耗时超标。

**验收**：全部目标测试通过；SIM 模式下小 shape 用例数值与 golden 一致。

### 阶段 9：文档与收尾

1. 新增/更新 `docs/zh/api/tensor_api/operation/pypto-conv_backward_input.md`（或 `pypto-conv_backward_filter.md`）、`docs/zh/api/tensor_api/config/pypto-set|get_convbp_*_tile_shapes.md`，并登记到对应 `index.md` 与 `docs/zh/api/index_hiascend.md`。
2. 更新 `docs/zh/guide/appendix/trouble_shooting/conv.md`：补新 opcode 的报错排查行。
3. 运行阶段 0 的检查脚本复验，全部模块 `present`。
4. 向用户报告：变更文件清单、测试结果、已知限制（首版不支持项）。

## 完成标准（Definition of Done）

- [ ] 阶段 0 检查脚本对目标算子输出全模块 `present`
- [ ] 全量构建通过（`python build_ci.py --no_iso --editable`）
- [ ] 框架 UT（operation impl + codegen）通过
- [ ] Python UT + ST（含 golden 对拍）通过
- [ ] API 文档与 trouble_shooting 更新
- [ ] 命名与 file-change-map 完全一致

## 常见陷阱

- **命名漂移**：opcode.cpp 注册的 TileOp 实现名（如 `TileOp::TLoad2DDX`）与 conv_bp_pto.h 中模板函数名不一致时，运行期才报找不到实现。阶段 1 的模块清单必须先冻结命名。
- **K0 语义混淆**：dx 的 K 块大小是 Cout0=16（K=Cout×Hk×Wk）；dw 的 N 内块是 Cin0=32B/dtype（fp16/bf16 下同为 16，但公式来源不同）。校验与 tile 对齐逻辑不要互换。
- **mL1/kL1 跨行**：dx 的 M 轴（HiWi）与 dw 的 K 轴（HoWo）都要求 tile 不跨 Wi/Wout 行（tileML1<Win 或整除；tileKL1<Wout 或整除），漏掉该约束会导致 Load3D/im2col 数据错位。
- **dw 的 CopyOut 时机**：必须在跨全部 batch 的 K 轴累加完成后才搬出，放在 batch 循环外、mL0 循环内。
- **L1 尾部脏数据**：HoWo 不满 16 对齐时，dw 的 Dy L1 搬入前必须清零整个 tile 区域，否则 MMAD 读到脏数据产生随机精度错误。
- **签名照抄与 -Werror**：框架侧 C++ 以 `-Werror=unused-parameter/-Werror=unused-variable` 编译。仿写同型函数（如 dx 的 `BpCheckAttrShape` → dw 的 `FilterCheckAttrShape`）时参数必须按新函数实际用途重新推导，不能整签名照搬——dw 把 Cout 一致性校验放在 `FilterCheckOutputShape` 后，`FilterCheckAttrShape` 就不再需要 tensor 参数。同理，会写回懒加载标志（aL1/bL1UpadateFlag）的调用链（`Construct*Tile` → `Construct*L1Tile`）整链的 `ConvBpIterInfo&` 必须为非 const，只改底层不改中间层会报 discards qualifiers。这类机械错误静态自查发现不了，提交前至少增量编译一次对应目标。
- **.so 双树**：bindings 改动后 pytest 可能加载旧 .so，报 `has no attribute` 时先 `python -c "import pypto.pypto_impl as i; print(i.__file__)"` 比对新旧。
