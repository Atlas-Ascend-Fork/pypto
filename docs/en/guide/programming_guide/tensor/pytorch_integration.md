# PyTorch Integration and Access

<!-- md-trans-meta sourceCommit=6c3d23fb55eb8bacdad9a9cfbcb9208870fc071b translatedAt=2026-10-27T02:39:52.516Z pushedAt=2026-10-27T08:54:55.572Z nl=2 -->

Currently, PyPTO supports two execution modes: single-operator mode (eager) and graph capture mode (aclgraph).

- Single-operator mode (eager): The code execution method is consistent with that of a normal Python program, where functions are executed immediately upon invocation without the need to build a compute graph. This mode is development- and debugging-friendly, but it introduces host task dispatch overhead. As performance optimization deepens, these host overheads gradually become a bottleneck and an issue that cannot be ignored.
- Graph capture mode (aclgraph): This mode uses the Capture&Replay approach to capture tasks once and replay them multiple times. During the capture phase, stream tasks are captured to the device side but not executed. During the replay phase, execution instructions are issued from the host side, and the device side then executes the captured tasks. This reduces host scheduling overhead and improves performance.

Adding the @pypto.frontend.jit decorator before a kernel function enables single-operator mode execution in the PyTorch framework by default. To enable graph capture mode, refer to the following code:

```python
B = pypto.DYNAMIC
N1, N2, DIM = 32, 1, 256

# enable frontend.jit for softmax
@pypto.frontend.jit()
def softmax_kernel(
    input_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32),
    output_tensor: pypto.Tensor((B, N1, N2, DIM), pypto.DT_FP32)
):
    ...



@allow_in_graph
def softmax(x: torch.Tensor, dynamic: bool = True) -> torch.Tensor:
    if isinstance(x, FakeTensor):
        return torch.zeros(x.shape, dtype=x.dtype, device=f'{x.device}')
    # launch the kernel
    out = torch.zeros(x.shape, dtype=x.dtype, device=f'{x.device}')
    softmax_kernel(x, out)
    return out


class MM(torch.nn.Module):
    def forward(self, x, dynamic):
        out = softmax(x, dynamic)
        return out


def test_softmax_capture(device_id=None, dynamic: bool = True) -> None:
    # prepare data
    ...
    model = torch.compile(MM(), backend="eager", dynamic=True)
    #graph capture
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y = model(x, dynamic)

    #execute graph
    g.replay()
    torch.npu.synchronize()
    ...

if __name__ == "__main__":
    test_softmax_capture()
```

You can view the info-level host compilation log and search for the **capture mode** keyword. The default value **0** indicates that graph capture mode is disabled, and **1** indicates that graph capture mode is enabled.

```text
2025-12-08 20:56:27.043    capture mode[1]
```

For the complete example, see [aclgraph.py](../../../../../examples/03_advanced/aclgraph/aclgraph.py).
