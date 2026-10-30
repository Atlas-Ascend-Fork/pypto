# 简介

SIMT程序中，不同线程的执行进度和内存访问完成顺序可能不同。当线程之间存在数据依赖时，需要使用同步屏障或内存栅栏约束执行顺序和内存可见性。

同步机制主要分为以下两类：

- **同步屏障（barrier）**：要求同一作用范围内的所有线程都到达指定位置后，线程才能继续执行。
- **内存栅栏（memory fence）**：约束调用线程在栅栏前后的内存访问顺序，使栅栏前的内存操作按指定范围对其他线程可见。内存栅栏不会等待其他线程到达同一位置。

## 同步屏障

pypto_pro.language.simt.syncthreads用于线程块内的阶段同步。以下以SIMD与SIMT混合编程的共享UB Tile为例，多个线程先写入数据，待所有线程完成写入后，再进入下一阶段读取这些数据。

```python
shared[0, tid] = value

# 等待当前线程块内的所有线程完成shared写入。
pl.simt.syncthreads()

result = shared[0, peer_tid]
```

每个线程都先写入自己的数据。到达syncthreads的线程会停止执行，直到当前线程块内的其他线程也到达同一个同步点。所有线程越过同步点后，才能安全读取其他线程在同步点前写入的共享数据。完整调用方式请参见[pypto_pro.language.simt.syncthreads](syncthreads.md)。

## 内存栅栏

内存栅栏用于约束调用线程的内存访问顺序。它通常解决“先写数据，再发布数据已就绪的标志”这类问题。

假设生产者线程依次执行以下两个写操作：

```python
data[0] = value
ready[0] = 1
```

代码中的先后顺序不代表其他线程一定按相同顺序观察到这两个写入。如果消费者线程已经观察到ready[0] == 1，但data[0]的新值尚未在相应作用范围内可见，消费者仍可能读取到旧数据。

在发布标志之前插入内存栅栏，可以建立所需的内存顺序：

```python
data[0] = value

# 保证前面的data写入先于后面的ready更新对其他线程可见。
pl.simt.threadfence()

pl.simt.atomic_exch(ready[0], 1)
```

上述代码建立的顺序关系如下：

```text
          data写入
             ↓
threadfence保证此前写入先行可见
             ↓
通过原子操作将ready更新为1
```

当消费者通过配套的同步协议观察到ready已经更新后，生产者在栅栏前写入的data也应当在对应作用范围内可见。

需要注意，内存栅栏只约束**调用线程自身**的内存访问顺序和可见性：

- 不会等待其他线程执行到栅栏位置。
- 不会通知消费者数据已经就绪。消费者仍需通过原子变量、轮询或其他同步方式判断状态。
- 不保证多个线程并发更新同一地址时的原子性。存在写入冲突时仍需使用原子操作。
- 不等同于线程屏障，不能替代syncthreads。

threadfence_block与threadfence的区别在于作用范围不同：

- threadfence_block用于当前线程块内的内存访问顺序约束，适合块内共享数据和UB协作场景。
- threadfence用于设备范围的内存访问顺序约束，适合通过GM向其他线程块发布数据或标志的场景。

接口的完整调用示例请参见[pypto_pro.language.simt.threadfence_block](threadfence_block.md)和[pypto_pro.language.simt.threadfence](threadfence.md)。
