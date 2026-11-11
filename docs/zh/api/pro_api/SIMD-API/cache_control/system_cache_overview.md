# 系统缓存概述

## Cache类型

Cache位于搬运单元或Scalar单元与GM之间，用于缓存频繁访问的数据和指令，降低访问延迟并减少对GM的重复读写。访问命中时，硬件直接从Cache获取数据或指令；未命中时，再从GM加载并更新对应Cache。下表介绍PyPTO Pro可配置或控制的Cache类型及其功能。

各类Cache与GM、搬运单元和计算单元之间的位置关系，参见[SIMD抽象硬件架构](../../../../guide/programming_guide/pro/programming_paradigm/SIMD/abstract_hardware_architecture.md#概览)中的图1。

| Cache类型 | 说明 |
|---|---|
| L2 Cache | 二级缓存，用于存储频繁访问的数据和指令，减少对GM的读写。<br>- MTE2读取GM数据时，优先从L2 Cache读取；未命中时再从GM加载，并更新L2 Cache。<br>- Scalar读取GM数据时，优先从DCache读取；DCache未命中时访问L2 Cache，L2 Cache也未命中时再从GM加载，并更新L2 Cache和DCache。<br>- 读取指令时，优先从ICache读取；ICache未命中时访问L2 Cache，L2 Cache也未命中时再从GM加载，并更新L2 Cache和ICache。 |
| Data Cache（DCache） | 用于缓存Scalar近期可能重复访问的数据。Scalar读取GM数据时，优先从DCache读取；DCache未命中时访问L2 Cache，L2 Cache也未命中时再从GM加载，并更新L2 Cache和DCache。 |
| Instruction Cache（ICache） | 用于缓存近期使用或频繁使用的指令。读取指令时，优先从ICache读取；ICache未命中时访问L2 Cache，L2 Cache也未命中时再从GM加载，并更新L2 Cache和ICache。 |

## 缓存控制接口汇总

| Cache类型 | 接口或参数 | 功能说明 |
|---|---|---|
| L2 Cache | [load](../memory_data_movement/load.md)、[load_tile](../memory_data_movement/load_tile.md)的l2_cache_mode | 配置GM搬入UB或L1 Buffer时的L2 Cache读取策略，类型为[TLoadL2Hint](../basic_data_structures/TLoadL2Hint.md)。 |
| L2 Cache | [store](../memory_data_movement/store.md)、[store_tile](../memory_data_movement/store_tile.md)的l2_cache_mode | 配置UB或L0C Buffer写回GM时的L2 Cache写入策略，类型为[TStoreL2Hint](../basic_data_structures/TStoreL2Hint.md)。 |
| DCache | [dcci](dcci.md) | 对指定地址对应的DCache执行清理并失效。 |
| ICache | [icache_preload](icache_preload.md) | 将调用位置之后的Kernel指令预加载到ICache。 |

各数据方向支持的具体L2 Cache策略，以相应接口的l2_cache_mode参数说明为准。
