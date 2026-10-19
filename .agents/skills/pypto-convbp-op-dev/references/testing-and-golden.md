# 测试与 golden 指南

## 1. golden 基准：torch.ops.aten.convolution_backward

卷积反向的标准 golden 与 aclnn 算子测试框架一致：

```python
torch.ops.aten.convolution_backward(
    grad_output,      # (N, Cout, Hout, Wout)
    input,            # (N, Cin, Hin, Win)——dx 需要前向输入构造；dw 即算子输入
    weight,           # (Cout, Cin, Kh, Kw)——dx 用真实 weight；dw 仅用 shape 推导
    bias_sizes=[],    # 无 bias
    stride=[sh, sw],
    padding=[ph, pw],
    dilation=[dh, dw_],
    transposed=False, output_padding=[0, 0], groups=1,
    output_mask=[...],   # dx: [True, False, False]；dw: [False, True, False]
)
```

- dx：先由 input_size 构造前向输入 x，调 aten 得 grad_input，与算子输出比对。
- dw：grad_output/input 为算子同款输入，weight 只参与 shape 推导（值不影响 grad_weight），取 grad_weight 比对。
- 精度阈值：`torch.allclose(out, golden, atol=1e-2, rtol=1e-2)`（FP16/BF16）。

## 2. kernel 编写模式（ST/demo 通用）

```python
def create_convbp_kernel(..., run_mode=pypto.RunMode.NPU):
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0},
        runtime_options={"run_mode": run_mode},
    )
    def kernel(grad_output: pypto.Tensor(grad_shape, dtype),
               weight_or_input: pypto.Tensor(..., dtype),
               out: pypto.Tensor(out_shape, dtype)):
        pypto.set_convbp_input_tile_shapes(l1_tile_info, l0_tile_info)   # dw 用 set_convbp_filter_tile_shapes
        pypto.set_vec_tile_shapes(...)                                    # 调用前必须设置
        output = pypto.conv_backward_input(                               # dw 用 conv_backward_filter
            grad_output, input_size_or_filter_size, weight_or_input, dtype,
            strides, pads, dilations, groups=groups)
        out.move(output)
    return kernel
```

- 运行模式：`--run_mode npu|sim`；SIM 用 cpu device，NPU 用 `torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))`。
- 输出 tensor 初始化为 `torch.zeros`，输入用 `torch.ones`（dx demo）或 `torch.rand`（dw demo）。
- tile 配置需满足 constraints-and-formulas.md 的全部公式，优先用 §6 的 demo 配置。

## 3. 测试文件组织

| 层 | 文件 | 内容 |
|----|------|------|
| 框架 UT（codegen） | `framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_mte/test_codegen_dyn_convbp.cpp` | `CodegenTestBase` 派生；`GenMockFuncDyn` 构造函数后逐 TileOp 加节点（SetAttribute + CopyOpAttribute），断言生成指令字符串包含 tileOp 名与参数 |
| 框架 UT（operation） | dx：`framework/tests/ut/operator/src/test_operation_impl.cpp`；dw：`framework/tests/ut/interface/src/operation/test_operation_impl_conv.cpp` | `TileShape::Current().SetConvBp*Tile(...)` + `SetVecTile` → `FUNCTION("...")` 内调 `ConvBp::ConvBackwardInput/ConvBackwardFilter`，校验图展开不报错 |
| Python UT | `python/tests/ut/operation/test_convbp_op.py` | 合法用例（fp16/bf16/dilation/no_pad/batch）+ 非法输入（不支持 dtype、dtype 不一致、非 2D、参数长度错、shape 关系不满足等）断言 PyptoError |
| Python ST | `python/tests/st/test_convbp.py` | NPU/SIM 双模式端到端 + golden 对拍（basic/stride/batch） |

## 4. 用例设计要求

1. **编译耗时门槛**：每个用例先算 tileTotalNum（公式见 constraints-and-formulas.md §4），门槛用例 <1000，禁止 >2000。
2. **非法输入覆盖**（Python UT 必测）：
   - dtype 不支持（如 int）/ 输入间 dtype 不一致 / out_dtype 与输入不一致
   - 维度非 4（dw）或 3/4/5 之外（dx）
   - strides/dilations/paddings/filter_size 长度不合法
   - dx：input_size 与 grad_output 不满足反向关系；dw：Hout/Wout 不满足正向关系
   - tile 约束违反（对齐、跨行、L1/L0 超限）→ 框架侧 CHECK 报错用例
3. **边界 shape**：cin/cout 非 16 整数倍（触发 CeilAlign 尾块）、houtL1Offset 为负（pad 区）、wout 尾行。
4. 运行方式：
   ```bash
   pytest python/tests/ut/operation/test_convbp_op.py -x -q
   pytest python/tests/st/test_convbp.py -x -q          # NPU 环境
   ctest --test-dir build/framework -R convbp           # 框架 UT（按仓内实际 target 名）
   ```
   批量跑 Python UT 用 `--forked`；C++ 改动后注意 .so 双树陷阱（见 SKILL.md 常见陷阱）。

## 5. 精度问题定位顺序

1. SIM 模式复现（排除硬件差异）。
2. 打开 `runtime_debug_mode`/`compile_debug_mode`，检查 TileGraph 展开日志（CONV_LOGW 的 offset/size 序列）。
3. 优先核对区间反算（constraints-and-formulas.md §5）与 pad 相对计算——绝大多数精度错误源于 houtL1Offset/winL1Offset 的 clamp 方向错误。
4. 核对 L1 清零是否遗漏（dw Dy 尾部脏数据呈随机性错误）。
5. 对照 docs/zh/guide/appendix/trouble_shooting/conv.md 的 opcode 排查表。
