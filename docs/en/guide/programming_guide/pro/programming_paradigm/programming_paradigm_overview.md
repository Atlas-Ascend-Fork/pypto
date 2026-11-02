# Programming Paradigm Overview

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-27T02:14:42.170Z pushedAt=2026-10-27T10:24:31.094Z nl=2 -->

PyPTO Pro uses Python syntax to write kernels that run on the NPU. Developers can describe data at different storage levels through Tensor and Tile, partition tasks using multi-core SPMD, and select SIMD, SIMT, or a combination of the two parallel approaches based on computation characteristics.

## Parallel Execution Models: SIMD and SIMT

To write high-performance device-side code, you must first understand the underlying parallel computing principles. In the field of high-performance parallel programming, SIMD and SIMT are two mainstream parallel execution models. They define the core mechanism by which instructions drive multiple computing units to work together, and are key technologies for improving program data throughput and optimizing computing performance. They are also the core content of learning PyPTO Pro programming.

The parallel execution process of PyPTO Pro:

1. The user launches a JIT Kernel on the Host side and configures the number of cores through `block_dim`.
2. Multiple cores execute the same Kernel program in SPMD (Single Program Multiple Data) mode, and process different data shards according to their respective core indexes.
3. Within each logical execution domain, the AIC mainly performs Cube-side SIMD cube computation, and the AIV mainly performs Vector-side SIMD or SIMT computation.
4. SIMD and SIMT can be combined in the same outer Kernel, and together with the data movement pipeline, complete the computation of an operator.

SPMD is responsible for organizing task parallelism among multiple logical AI Cores, while SIMD and SIMT are responsible for describing the computation method within a single logical execution domain. The two operate at different levels.

## SIMD (Single Instruction Multiple Data)

SIMD (Single Instruction Multiple Data) is a data-parallel model. Its core logic is that a single instruction performs exactly the same operation on multiple data elements within the same clock cycle, achieving batch parallel processing of data.

PyPTO Pro uses Tensor to describe data in GM and Tile to describe data blocks in on-chip memory. Developers use Tile or Vector Register as the primary programming objects, without needing to describe the same computation for each data element individually.

### Core Characteristics

- **Single instruction driven**: All parallel computation units execute the same instruction synchronously, with completely identical operations;
- **Homogeneous data**: The data involved in computation must be of a unified type and equal length, ensuring that instructions can process them in batches;
- **Synchronous execution**: Operations on all data are completed within the same instruction cycle, with no independent scheduling logic and a fully unified execution rhythm.

### Scenarios

SIMD is suitable for continuous, regular data access and scenarios where the same operation is performed on a large amount of data, for example:

- Element-wise computation, reduction, and data transformation;
- High-density computation such as cube multiplication;
- Fused computation jointly performed by Cube and Vector;
- Core computation paths that require high throughput and hardware resource utilization.

A typical SIMD Kernel is organized through the process of tiling design, data loading, data computation, and data unloading. For details, see [SIMD Programming Paradigm](SIMD/programming_paradigm.md).

## SIMT (Single Instruction Multiple Threads)

SIMT (Single Instruction Multiple Threads) is a thread parallel model. The same SIMT instruction is executed in parallel by multiple independent threads. Each thread processes different data through its own index, and can independently perform address computation and conditional branching.

### Core Characteristics

- **Multi-thread parallelism**: The same program is executed by multiple threads in parallel, with each thread processing different data.
- **Thread independence**: Each thread has its own index and execution state, can address memory independently, and supports branches and loops.
- **Parallel scheduling**: Multiple threads are scheduled in parallel by hardware, and different threads can take different control paths.

SIMT programming first defines a SIMT entry function, and then the outer JIT Kernel launches a Thread Block in the Vector execution domain.

### Scenarios

SIMT is suitable for scenarios that require per-thread control or are difficult to express with regular Tile computation, for example:

- Discrete indices and irregular data access;
- Computation with many branches or where different data requires different processing logic;
- Computation that requires atomic updates, such as histograms and counters;
- Sparse data processing and local computation with strong dynamic data dependencies.

A typical SIMT computation first defines a SIMT function, partitions data by thread index, and then uses scalar computation, synchronization, or atomic APIs to complete per-thread processing. For details, see [SIMT programming paradigm](SIMT/programming_paradigm.md).

## AI Core Hardware Fundamentals

An AI processor contains multiple AI Cores, which can process different data shards in parallel. Each AI Core contains units such as Scalar, Vector, Cube, on-chip memory, and data movement:

| Hardware Composition | Primary Responsibility | Corresponding Expression in PyPTO Pro |
|---|---|---|
| Scalar unit | Kernel control flow, address computation, and instruction scheduling | Python control flow, scalar expressions, and data index computation |
| Vector unit | Vector computation and SIMT thread computation | Tile APIs, VF functions, and SIMT functions in the Vector execution domain |
| Cube unit | Cube computation such as cube multiply-accumulate | Cube computation APIs in the Cube execution domain |
| On-chip memory | Stores computation inputs, outputs, and intermediate data | Tiles in different memory spaces |
| Data movement unit | Moves data between GM and on-chip memory, and between different on-chip memories | Data loading, on-chip movement, and result write-back operations |

Computation and data movement tasks execute on different pipelines. When data dependencies exist, synchronization mechanisms are required to constrain the execution order.

For the mapping of hardware units, memory hierarchy, and their APIs, see [SIMD Abstract Hardware Architecture](SIMD/abstract_hardware_architecture.md) and [SIMT Abstract Hardware Architecture](SIMT/abstract_hardware_architecture.md).

## Development Process and Learning Path

The typical development and running process of a PyPTO Pro operator is as follows:

1. On the Host side, use PyTorch to prepare the input, output, and workspace tensors.
2. Use the JIT decorator to define the Kernel, and declare parameters such as Tensor, Scalar, or TilingData.
3. Design multi-core partitioning and tiling based on the data scale, and plan the tiles and on-chip memory.
4. Select SIMD, SIMT, or combine both computation methods in the same Kernel based on the data access pattern and control flow characteristics.
5. Launch the Kernel on the Host side, and configure the Stream and `block_dim` as needed; the first call triggers JIT compilation.
6. After the Kernel is dispatched asynchronously, wait for the Device task to complete before reading the result on the Host side.

It is recommended to continue reading the following:

- [PyPTO Pro Quick Start](../../../quick_start/pro/index.md): Learn the Kernel definition, compilation, and running process through a complete operator example.
- [SIMD Programming](SIMD/index.md): Learn about SPMD multi-core parallelism, Tile programming, SIMD vector computation, and cube computation.
- [SIMT Programming](SIMT/index.md): Learn about the thread architecture, memory hierarchy, SIMT functions, synchronization, and atomic operations.
- [Kernel Function](../development/kernel_function.md): Learn about Kernel parameters, the actual number of cores, and the launch method.
- [Compilation and Execution](../development/compilation_and_execution/index.md): Learn about JIT compilation and offline binary compilation.
