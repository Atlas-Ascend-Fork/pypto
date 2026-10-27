# 抽象硬件架构

PyPTO Pro的SIMT函数运行在AIV（Vector Core）上。

**图1 SIMT抽象硬件架构**

![GM、L2 Cache与多个Vector Core的关系](../../../../figures/pro/simt_abstract_hardware_architecture.svg)

AI处理器内部有多个Vector Core，每个Vector Core包含计算单元、Shared Memory（位于UB）和寄存器。核外的GM是全局内存空间，被所有Vector Core共享。L2 Cache位于GM与各Vector Core之间，也由多个Vector Core共享。

SIMT计算涉及的主要硬件资源如下：

- **计算单元**：执行SIMT函数中的标量计算、地址计算和控制逻辑。
- **寄存器和栈空间**：每个Thread独立使用，用于保存函数参数、线程索引、局部变量和中间结果。
- **Shared Memory**：使用UB中的部分空间，供同一Thread Block内的Thread交换数据和复用中间结果。
- **Data Cache**：使用UB中的部分空间，缓存SIMT线程访问的GM数据。
- **L2 Cache**：缓存GM数据并降低访问延迟，由硬件管理。
- **GM**：保存输入、输出以及不同Thread Block访问的数据。

SIMT线程访问GM时，数据经过L2 Cache和Vector Core内的Data Cache后进入Thread私有寄存器。Thread Block内共享的数据可以保存在Shared Memory中，由块内Thread直接访问。
