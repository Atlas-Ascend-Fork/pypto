# Softmax Operator Quick Start

<!-- md-trans-meta sourceCommit=a82d13abf11a8290f5bf9637ac25356dcc37f489 translatedAt=2026-10-27T02:43:27.944Z pushedAt=2026-10-27T08:52:06.688Z nl=2 -->

## Task and Objective

This section guides you through implementing a simple Softmax operator using the PyPTO framework and verifying its correctness with test cases. After reading this section, you will understand how to use PyPTO APIs to build custom operators. Once the program is complete, you can also use the PyPTO Toolkit visualization tool to view the compute graph structure and observe various performance metrics of the operator.

The example code for the Softmax operator implementation is located at: [softmax.py](../../../../../examples/02_intermediate/operators/softmax/softmax.py). You can refer to this code example to better understand the content of this section.

## Operator Design Specification

**Table 1** Softmax operator design specification

| name                | shape            | data type | format |
| ------------------- | ---------------- | --------- | ------ |
| **inputs**  | (-1, 32, 1, 256) | float     | ND     |
| **outputs** | (-1, 32, 1, 256) | float     | ND     |

* Mathematical expression

  $M = \max(z),\quad \text{SoftMax}(z_i) = \frac{\exp(z_i - M)}{\sum_j \exp(z_j - M)}$

* Main APIs used

  Basic computation APIs: exp, sum, / (div), amax, - (sub)

## Importing PyPTO Modules

Before implementing the Softmax operator, you need to import the PyPTO, PyTorch, and Numpy modules. The PyPTO module provides tensor operations and compilation capabilities, while the PyTorch and Numpy modules are used for result verification.

```python
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
```

## Core Code Logic

1. Implement the core computation function.

    PyPTO provides a rich set of Operation APIs for implementing different computation logic. Developers can combine different Operation APIs based on the operator's mathematical expression to implement complex computation logic. The following is the core computation function implementation of the Softmax operator:

    ```python
    def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
        row_max = pypto.amax(x, dim=-1, keepdim=True)  # Compute the row maximum
        sub = x - row_max                              # Normalize the values
        exp = pypto.exp(sub)                           # Perform the exponential operation
        esum = pypto.sum(exp, dim=-1, keepdim=True)    # Compute the sum
        return exp / esum                              # Normalize the probabilities
    ```

2. Implement the Softmax kernel function.

    To enable efficient execution of the computation logic on hardware, you need to implement the Softmax kernel function, use the @pypto.frontend.jit decorator to convert the compute graph into hardware instructions, and define strategies such as data tiling and loop processing within it. When calling the function, you can directly pass a PyTorch tensor, and the PyPTO framework will automatically handle tensor type conversion.

    ```python
    @pypto.frontend.jit
    def softmax_kernel(
        input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
        output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    ):
        bs, seqlen, head, dim = input_tensor.shape
        tile_b = 1  # Process one batch at a time
        b_loop = bs // tile_b

        # Tiling shape setting for efficient execution
        pypto.set_vec_tile_shapes(1, 4, 1, 64)

        for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
            b_offset = idx * tile_b
            b_offset_end = (idx + 1) * tile_b
            input_view = input_tensor[b_offset:b_offset_end, :seqlen, :head, :dim]
            softmax_out = softmax_core(input_view)
            output_tensor[b_offset:b_offset_end, ...] = softmax_out

    ```

    To improve the computation efficiency of the operator, you can use the set\_vec\_tile\_shapes or set\_cube\_tile\_shapes API to specify the tiling method for the operation. This tiling configuration decomposes the computation into hardware-friendly tile granularity (such as 64), which optimizes memory access and parallel computation efficiency.

    ```python
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    ```

## Test Case

To verify the correctness of the Softmax operator, write a test case. This test case uses a PyTorch Tensor as input, performs computation through the PyPTO kernel, and compares the result with that of PyTorch's built-in Softmax function. Before executing PyPTO and PyTorch related code, you need to specify the corresponding device ID, or obtain the current device ID through the torch.npu API.

```python
def test_softmax(device_id: int = None, run_mode: str = "npu", dynamic: bool = True) -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    x = torch.rand(shape, dtype=torch.float, device=device)
    y = torch.zeros(shape, dtype=torch.float, device=device)

    softmax_kernel(x, y) # default dim: -1
    golden = torch.softmax(x, dim=-1).cpu()
    y = y.cpu()

    max_diff = np.abs(y.numpy() - golden.numpy()).max()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(y), np.array(golden), rtol=3e-3, atol=3e-3)
    print("✓ Softmax test passed")
    print()
```

## Compilation and Execution

Switch to the directory where the sample code is located and run the following in an environment where PyPTO is installed:

```bash
# Configure the CANN environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the device ID
export TILE_FWK_DEVICE_ID=0

# Execute the script
python3 softmax.py
```

After the program is successfully executed, the following information is displayed:

```text
Input shape: torch.Size([32, 32, 1, 256])
Output shape: torch.Size([32, 32, 1, 256])
✓ Softmax test passed
```

Meanwhile, compilation and execution result files are generated in the $\{work\_path\}/output/output\_\*/ (* represents the timestamp).

## Viewing Compute Graphs

During compilation, a PyPTO program automatically generates a graph structure composed of tensors and Operations, known as the compute graph. This compute graph goes through the PyPTO compilation optimization process, completing the compilation from the original compute graph to an executable graph, and ultimately producing executable code that can run on the Ascend hardware environment to perform actual computation tasks. You can use the PyPTO Toolkit visualization tool to view key information in the compute graph.

1. Right-click the $\{work\_path\}/output/output\_\*/program.json file and choose **Open with PyPTO Toolkit** from the pop-up menu.

    The program.json file contains summary information of the Execute Graph and Block Graph. The key information in the graph is as follows: the cards on the left and right sides are tensor nodes (representing input/output data), and the middle card is a call node (marked with an fx identifier; click it to drill down for more details).

    ![](../../figures/tensor/graph_1.png)

2. Double-click the middle card to drill down layer by layer to the Execute Graph shown in the following figure.

    ![](../../figures/tensor/graph_2.png)

    The different color blocks (CALL:TENSOR\_xx) in the figure each represent a call node, indicating that the compute graph is divided into different Block Graph subgraphs.

3. Double-click the call node in the figure above to view the Block Graph subgraph information, which identifies the specific execution process of the task.

    ![](../../figures/tensor/graph_3.png)

    After zooming in, you can see the specific tensor and Operation node information and connection relationships:

    ![](../../figures/tensor/graph_4.png)

## Viewing the Swimlane Diagram

The swimlane diagram visually displays the actual scheduling and execution process of the compute graph, clearly presenting the execution order and time consumption of tasks, helping developers analyze operator performance bottlenecks. This section describes how to collect swimlane diagram data and view the swimlane diagram through PyPTO Toolkit.

1. Enable the performance data collection feature by configuring the graph execution phase debug switch through the `debug\_options` parameter of the `@pypto.frontend.jit` decorator.

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

2. Run the operator program again.

    ```bash
    python3 softmax.py
    ```

    A swimlane diagram data file named `merged\_swimlane.json` is generated in the $\{work\_path\}/output/output\_\*/ directory (where \* represents a timestamp).

3. View the swimlane diagram via the PyPTO Toolkit plugin.

    Right-click merged\_swimlane.json and choose "Open with PyPTO Toolkit" from the pop-up menu, as shown in the following figure.

    **Figure 1**  Swimlane diagram UI
    ![](../../figures/tensor/swimlane_graph.png "Swimlane diagram UI")

    The colored blocks in the figure above are swimlanes, showing the task execution status on each AIC/AIV. The length of a swimlane entry corresponds to the task duration, providing an intuitive view of computation intensity. You can analyze potential performance bottlenecks by observing idle gaps between adjacent swimlanes (the black areas in the figure, also known as bubbles) and swimlane entries with long durations.
