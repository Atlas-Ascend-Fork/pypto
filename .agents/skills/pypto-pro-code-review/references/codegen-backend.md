# CCE codegen / backend 检视规则

## 适用范围

IR 到生成代码的转换、op 注册、指令映射及后端约束的改动。
涉及 IR 语义、目标架构选择或源码编译执行时，通过[模块索引](review-checklist.md)
加载 IR、runtime 等相关模块规则。代码结构分析、优化、变换遵循 [Pass 规则](pass.md)。

## 源码入口

- [Codegen 声明](../../../../framework/include/pypto_pro/codegen/)
- [Codegen 实现](../../../../framework/src/interface/pypto_pro/codegen/)
- [Backend 声明](../../../../framework/include/pypto_pro/backend/)
- [Backend 实现与 op 注册](../../../../framework/src/interface/pypto_pro/backend/)
- [PyPTO Pro Python binding](../../../../python/src/bindings/pypto_pro/)
- [Codegen 单元测试](../../../../python/tests/ut/pypto_pro/codegen/)

## 模块专属规则

### PRO-CODEGEN-001：op 声明、实现与返回类型一致

- **适用条件**：新增或修改 op 的后端声明、实现或 IR 到 CCE 的转换。
- **检查内容**：暴露给前端的 op 按公共规则 `PRO-API-001` 核对完整跨层一致性。
  仅供框架内部使用的 IR call op 允许只在后端定义，核对实际构造、调用方与后端声明、实现的
  参数、返回值及语义是否一致，不要求补充前端层。所有 op 均不得漏加或多加参数及返回值；
  无返回值 op 的返回类型必须设为 `NoneType`。
- **报告证据**：具体 op 在其实际适用层之间的契约差异，或无返回值 op 的错误返回类型；
  内部 IR call op 缺少前端公开接口本身不构成违规。
- **用户依据**：codegen 1；通用 1.1、5.3。

tuple 类型比较遵循公共规则 `PRO-TUPLE-001`：结构相等不能替代类别信息比较。
新增暴露给前端的 op 还须加载 [DSL/parser 规则](dsl-parser.md)，检查参数解析与 `@op_impl` 注册。
