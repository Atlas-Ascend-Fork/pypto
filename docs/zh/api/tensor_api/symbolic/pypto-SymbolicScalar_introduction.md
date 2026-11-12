# pypto.SymbolicScalar简介

SymbolicScalar用于实现数据的动态表达，比如Tensor运行时实际传入的Shape、Tensor运行时实际的数值，以及基于运行时数据产生的算术运算表达。

## 类型别名SymInt

SymInt是类型别名，等价于Union[int, SymbolicScalar]，用于标注那些既接受Python整数、又接受SymbolicScalar的参数。它出现在pypto.loop、pypto.loop_unroll、pypto.cond、pypto.is_loop_begin、pypto.is_loop_end、Tensor.shape、pypto.tril、pypto.triu等接口的类型签名中。

SymInt仅用于类型标注，不是可调用的运行时对象：

- 传入int：走静态值路径，用于编译期即可确定的取值，例如固定的循环次数或固定的对角线偏移。
- 传入SymbolicScalar：走符号化路径，取值在运行时才确定，例如pypto.loop返回的循环索引，或由张量shape构造的表达式。
