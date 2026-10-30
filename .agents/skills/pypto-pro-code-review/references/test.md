# Test 检视规则

## 适用范围

PyPTO Pro 及相关公共 IR 的 UT/ST 用例、golden 文件和影响以下规则的测试辅助逻辑。
沿被测接口和测试 helper 核实执行与断言行为；只报告本次引入或扩大的问题。
测试改动按本文件检视，不因调用某个框架模块就将该模块的实现规则套到测试代码上。

## 源码入口

- [Pro UT](../../../../python/tests/ut/pypto_pro/) 与 [Pro ST](../../../../python/tests/st/pypto_pro/)
- [完整 IR 文本对比用例](../../../../python/tests/ut/pypto_pro/ir/test_function_type_ir/test_function_type_python_print.py)
  及其 [.pypto golden](../../../../python/tests/ut/pypto_pro/ir/test_function_type_ir/test_function_type_python_print.pypto)
- [完整文本对比与增强校验用例](../../../../python/tests/ut/pypto_pro/runtime/test_parse_observer_ir/test_observer_sees_final_program_once.py)
- [公共 IR snapshot helper](../../../../python/tests/ut/ir/test_common.py)

## 测试专属规则

### PRO-TEST-001：按是否在 NPU 上执行划分 ST 与 UT

- **适用条件**：新增测试用例，或修改用例的执行路径、目录及测试分类。
- **检查内容**：ST 用于 NPU 上板执行用例，必须存在被测逻辑在 NPU 上实际执行的路径；
  非上板用例应定义为 UT，并放入对应 UT 目录。只进行 parser/IR 构建、Pass 变换、源码生成、
  编译或 mock 执行的用例都属于 UT，不能因覆盖多个模块或调用 JIT 接口就归为 ST。
  沿 fixture、helper、编译与 launch 调用确认实际路径；NPU 标记、导入 `torch_npu`、
  查询设备或初始化设备本身不能证明被测逻辑上板执行。
- **报告证据**：用例位置及分类、被测逻辑的实际执行链、是否存在 NPU 执行，以及应归属的目录。
- **边界**：真实上板用例因设备不可用或 SoC 不匹配而跳过，仍属于 ST；不能仅根据一次运行被
  skip 判为 UT。验证记录是否足以证明已上板，另按 `PRO-VALIDATE-001` 判断。

### PRO-TEST-002：IR 与源码字符串断言使用完整 `.pypto` golden

- **适用条件**：UT 新增或修改对 IR 打印结果、生成源码等文本内容的断言，或修改对应 golden
  及比较 helper。
- **检查内容**：参考已有 UT，将预期的完整目标文本保存为 `.pypto` 文件，读取该 golden，
  与实际输出的完整 IR 或源码作一致性比较。检查 golden 是否实际参与断言，不能只创建文件。
  仅用子串包含/排除、正则、出现次数或抽取若干行来验证文本均不满足要求；在 Python 中
  内嵌若干预期片段也不能替代 `.pypto` golden。
  可沿用现有用例对末尾换行的处理，但不能删除语句、只保留关注行等来缩小比较范围。
  在完整文本对比基础上，可追加关键内容、次数、顺序、IR 节点类型或其他行为的增强校验；
  这些断言不能替代完整对比。
- **报告证据**：实际文本的生成位置、`.pypto` golden 路径及内容范围、完整比较调用链，
  或仅有片段断言的位置与未覆盖内容。完整比较位于 helper 中时应沿调用确认，避免误报增强断言。
- **边界**：完整文本对应被测输出对象，如函数、Program 或生成源文件，不要求拼入无关输出。
  本规则针对 IR/源码文本一致性，不将异常消息匹配或普通标量属性断言强行改为 golden 测试。
