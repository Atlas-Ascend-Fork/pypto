# DSL / parser 检视规则

## 适用范围

用户 DSL 语法、解析、作用域、类型解析及从 DSL 构建 IR 的改动。
parser 只承担构建 IR 所依赖的必要优化和 lowering（如 SSA、常量折叠、函数展开）；
其他代码结构分析、优化、变换遵循 [Pass 规则](pass.md)。
若改动影响 IR 语义或后续生成代码，通过[模块索引](review-checklist.md)继续加载相关模块规则。

## 源码入口

- [DSL 与 parser](../../../../python/pypto_pro/language/)
- [表达式解析](../../../../python/pypto_pro/language/parser/_expression_parser.py)
- [调用与参数解析](../../../../python/pypto_pro/language/parser/_call_parser.py)
- [op_impl 注册入口](../../../../python/pypto_pro/ir/op/_op_registry.py)
- [DSL 与 parser 单元测试](../../../../python/tests/ut/pypto_pro/language/)

## 模块专属规则

### PRO-DSL-001：复用通用 AST 到 IR 的解析能力

- **适用条件**：新增 DSL 语法、AST 节点处理或解析场景。
- **检查内容**：先确认已有 parse 场景是否能够覆盖新功能；明确不能覆盖时，梳理新功能依赖的
  AST 节点场景及其到 IR 的 lowering 方式，考虑与其他 AST 结构嵌套时的行为。
  避免仅为单个语法外形另写一套已有解析逻辑。
- **报告证据**：可复用的解析入口，或不能正确处理的新 AST 场景、嵌套结构及其预期 IR。
- **用户依据**：DSL/parser 1；通用 5.1。

### PRO-DSL-002：基于 IR node 的 type 校验类型

- **适用条件**：新增或修改类型相关的校验。
- **检查内容**：所有类型相关校验应基于解析后的 IR node 的 `type`，充分复用框架的 parse 能力；
  不以 AST 的表面语法或自行推断结果替代 IR 类型。tuple 类型比较同时遵循公共规则 `PRO-TUPLE-001`。
- **报告证据**：具体类型检查位置、对应 IR node/type，以及绕过现有解析能力的逻辑。
- **用户依据**：DSL/parser 2–3。

### PRO-DSL-003：attribute 与表达式参数分离

- **适用条件**：新增暴露给前端的 op 或修改其前端参数解析方式；不适用于仅供框架内部使用的 IR call op。
- **检查内容**：attribute 参数只能是 Python 常量值，只能通过 keyword arguments 传递，
  使用 `parse_op_kwargs` 解析，即先经过 `parse_expression` 再转为 Python 常量值。
  其余参数只能通过 positional arguments 传递，并使用 `parse_expression` 转成 IR expr。
  按公共规则 `PRO-ERROR-001` 校验参数约束，不重复实现解析或校验逻辑。
- **报告证据**：API 中的参数类别、实际传参方式、解析调用及最终值类型。
  仅调用 `parse_op_kwargs` 不足以证明 attribute 合规，还须确认结果满足 Python 常量约束。
- **用户依据**：通用 1.5。

### PRO-DSL-004：新增前端 op 的注册与通用分发

- **适用条件**：新增 op 的前端解析实现，或在通用分发中添加 op 专属分支。
- **检查内容**：通过 `@op_impl` 装饰器接入，不使用 `register_table` 或其他自行注册方式。
  仓内已有的其他注册方式不构成新增 op 的例外；后端声明与实现仍须按公共规则 `PRO-API-001` 对齐。
  对公共 parser 按 op name 新增的拒绝分支，沿注册表、默认处理、原生 op 构造及异常传播追踪：
  不加该分支时，同一非法调用是否已由通用机制拒绝？若是，不应为新 op 枚举特殊报错；
  op 特有约束放在对应处理入口，通用层处理共性规则。与 `PRO-SIMPLE-001` 同根因合并报告。
  必须检查默认路径，不能将“未注册 Python hook”直接等同于“不支持”，因为原生 op 仍可能处理它。
- **报告证据**：新增 op 的解析处理函数、注册路径，以及专用报错分支对应的通用拒绝链和异常契约。
  拒绝发生得更早或文案更具体，本身不足以证明需要给公共框架添加 op 名单。
- **反例**：通用路径尚未覆盖的独立输入约束、必须阻止副作用的边界检查，或已明确要求的诊断契约，
  需核实后保留在恰当层；不能将所有提前校验或所有按 op 分发的合法处理一概判为冗余。
- **用户依据**：通用 1.6；DSL/parser 4；[PR 6992](https://gitcode.com/cann/pypto/pull/6992/)
  评论 `195268837`：`_call_parser.py:1045` 的 GEMM 专用报错；对应 head
  `f9e6350fc653730e210f60d9b9f426f2f2f448a4` 中默认处理最终由 `OpRegistry::Create` 拒绝未注册 op。
