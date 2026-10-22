# Pass 检视规则

## 适用范围

源码或 IR 的代码结构分析、优化、转换，以及 Pass 的实现和接入。
按功能职责选择本模块，不能因实现位于 parser、codegen 或 runtime 目录就跳过。
parser 构建 IR 所依赖的必要处理边界见[通用清单的模块职责](review-checklist.md)。

## 源码入口

- [Pass、PassImpl 与工厂接口](../../../../framework/include/ir/transforms/passes.h)
- [Pass 基础实现](../../../../framework/src/interface/ir/transforms/passes.cpp)
- [IR 变换实现](../../../../framework/src/interface/ir/transforms/)
- [Pass Python binding](../../../../python/src/bindings/ir/pass.cpp)

这些入口用于理解和查证框架接口，不表示现有分析或转换实现均符合用户规则。

## 模块专属规则

### PRO-PASS-001：基于框架 Pass 体系实现分析与变换

- **适用条件**：新增源码或 IR 的分析、优化、转换功能，或新增、修改 Pass 实现。
- **检查内容**：功能应实现为 Pass，基于框架 Pass 基类体系构造或继承其实现基类，
  不另写一个任意类来替代 Pass 做 IR 转换。核实该功能在框架流程中的接入位置和调用方式。
  当前框架的 `CreateFunctionPass`、`CreateProgramPass` 工厂以及 `PassImpl` 实现类都是
  基于 Pass 体系的接入方式，不要求每个 Pass 都另建派生类。
  默认按变换粒度选择 Function/Program 工厂；需要自定义状态时再使用 `PassImpl` 实现类。
- **报告证据**：该功能承担的分析或变换职责、实际实现与调用路径，以及绕过 Pass 体系的位置。
  Pass 内部使用 visitor、mutator 或辅助函数不等于绕过 Pass；应检查对外的功能入口。
- **用户依据**：通用 2.2、5.2；pass 1。

涉及基础 IR 节点或类型变化时继续读取 [IR 规则](ir.md)；模块独立性遵循公共规则 `PRO-MODULE-001`。
