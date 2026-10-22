# 前端诊断与 IR 观测

编译期错误中的源码位置指向用户 Kernel。错误包含类型、错误码、原因和可选 hint；源文件可读取时还会显示附近源码与标记。Tile 地址、容量、有效范围和搬运路径检查使用结构化位置，而不是把位置拼接进原因文本。

直接调用 `make_tile_expr` 或 `_ir_reinterpret` 时，地址、容量、对齐和 buffer 大小的错误及 warning 优先使用显式提供的 `span`；未提供时继承当前 `current_span` 上下文的位置。

## 编译期 warning

warning 继承 `pypto_pro._errors.PyptoProWarning`，可使用 Python warning filter 控制。默认启用的 `CapacityUnknown` 表示某个有固定容量的片上 buffer 无法查询容量，检查因此跳过；应核对目标架构、安装的 native binding 和 platform ini。MX scale 的 `ScaleLeft/ScaleRight` 同样有固定片上容量，查询失败时也会报警。DDR 不属于固定片上 buffer，不触发这类 warning。

`MutexMismatch` 表示手写 mutex ID 的 TileGroup 槽地址重叠，但没有共享 ID，`auto_mutex` 无法通过这些锁为同一段字节建立顺序。该 warning 仅在 `auto_mutex=True` 时检查，复用原生 TileGroup 元数据；AUTO 分配已有的冲突拒绝先执行。不重叠的地址可以复用手写 ID，不因此报警。

动态偏移是合法写法，因偏移不是编译期常量而跳过的负值和边界检查默认保持安静。需要排查未检查的内容时，设置 `PYPTOPRO_INFO=skipped-checks` 或 `PYPTOPRO_INFO=all`，会产生带 Kernel 位置的 `CheckSkipped` warning；这不表示动态偏移已被证明安全。

```bash
PYPTOPRO_INFO=skipped-checks python kernel.py
python -m pytest -W error::pypto_pro._errors.CapacityUnknown
```

Tile 容量检查在构建时执行，第一处超限立即抛出 `OutOfRange` 并终止当前解析。正常编译解析和 pipeline 探测解析均使用这一行为，错误位置指向超限的 Tile 声明。

TileType 的结构相等性同时检查有效 MemRef 是否存在和 memory space 是否一致。两侧均未指定 MemRef 时可以相等；仅一侧指定时不相等。分配地址和大小不参与类型比较。

JIT 在编译绑定和 launch 时使用相同的 Tensor dtype 检查。FP4、HF4、INT4 和 UINT4 使用打包的 `torch.uint8` carrier；FP4E2M1 还接受当前 Torch 提供的 `torch.float4_e2m1fn_x2`。FP8E4M3FN、FP8E5M2 和 FP8E8M0 接受各自对应的 Torch dtype 或原始 `torch.uint8` carrier，不混用不同 FP8 格式。INT16 和 UINT16 均接受 `torch.int16` 或 `torch.uint16` carrier；UINT32 接受 `torch.uint32` 或 `torch.int32` carrier。相同位宽的整数 carrier 按原始位模式传入，不进行数值转换。其他 Tensor 参数使用与声明对应的 Torch dtype；检查只验证存储类型，不进行数值转换。4-bit 参数的宿主末维按两个逻辑元素／存储单元换算，显式 `static_shapes` 已是逻辑元素数。

## 诊断断言

`str(error)` 保留源码预览。测试中可先用 `pytest.raises(...) as caught` 捕获异常，再对 `caught.value.message` 断言错误原因，避免匹配到源码预览中的文本。需要同时检查错误码、位置、hint 和 note 时，显式检查 `caught.value.render(with_source=False)` 的结果。

## 只读 IR attribute 快照

`pypto_pro.ir.get_attributes(node)` 支持 Call、Function 和 ForStmt，返回完整的受支持 attribute 字典，包括列表与表达式元数据。未知 attribute 类型会带字段名抛出 TypeError，不静默丢弃。容器为副本，表达式值仍引用原生 IR 对象；观测时不得修改它们。

```python
from pypto_pro import ir

if getattr(ir, "attributes_api_version", None) != 1:
    raise RuntimeError("This reader requires attributes API version 1")
attributes = ir.get_attributes(node)
```

内部 `KernelDef.parse_target_program(..., observer=...)` 提供本次调用局部的 `observer(program, parser)`，在 AUTO mutex ID 分配后、返回最终 Program 前调用。parser 的辅助表保留解析期元数据，不能假定其中的 AUTO ID placeholder 已随 Program 一起替换。observer 异常传给调用方，不保留为后续调用的回调；该内部入口需要针对实际安装版本进行能力探针。
