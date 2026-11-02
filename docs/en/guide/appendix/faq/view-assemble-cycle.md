# Performing View and Assemble on the Same Tensor Causes a Graph Cycle Error

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:21:22.830Z pushedAt=2026-10-27T02:49:13.071Z nl=2 -->

## Symptoms

The following shows example code:

```python
@pypto.frontend.jit
def foo_kernel(x, y):
    pypto.set_vec_tile_shapes(16, 16)
    a = pypto.zeros([32, 32])
    b = a[:16, :16]          # View data from a
    a[16:, 16:] = b.exp()    # Assemble and write back to a after computation
    y[:] = x + a

torch.npu.set_device(0)
x = torch.ones(32, 32, dtype=torch.float32)
y = torch.empty(32, 32, dtype=torch.float32)
foo_kernel(pypto.from_torch(x), pypto.from_torch(y))
```

An error is reported during execution: `ASSERTION FAILED`

```text
ERROR:root:Record function foo_kernel failed: ASSERTION FAILED: outDegree[opToIndex[op.get()]] == 0
```

The detailed error message is as follows:

```text
ERROR:root:Record function foo_kernel failed: ASSERTION FAILED: outDegree[opToIndex[op.get()]] == 0
Operation not fully processed: /* /home/pypto-dev/a.py:9 */
<32 x 32 x DT_FP32 / 32 x 32 x DT_FP32> %0@2#(-1)MEM_UNKNOWN::MEM_UNKNOWN = !10000 VEC_DUP(g:-1, s:-1) #SCALAR{0.000000} #op_attr_shape{[32, 32]} #op_attr_validShape{[32,32]}
, func GetSortedOperations, file function.cpp, line 1105
libtile_fwk_interface.so(npu::tile_fwk::Function::GetSortedOperations() const+0xb3c) [0xffff9c2f6650]
libtile_fwk_interface.so(npu::tile_fwk::Function::SortOperations()+0x38) [0xffff9c2f6f28]
libtile_fwk_interface.so(npu::tile_fwk::Function::EndFunction(std::shared_ptr<npu::tile_fwk::TensorSlotScope> const&)+0x960) [0xffff9c31b8d0]
libtile_fwk_interface.so(npu::tile_fwk::Program::FinishCurrentFunction(std::shared_ptr<npu::tile_fwk::TensorSlotScope> const&, bool)+0x1b0) [0xffff9c532274]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndFunction(std::string const&, bool)+0x10c) [0xffff9c536dcc]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndHiddenLoop(npu::tile_fwk::Function*, bool)+0xb0) [0xffff9c537384]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndFunction(std::string const&, bool)+0x5c) [0xffff9c536d1c]
libtile_fwk_interface.so(npu::tile_fwk::RecordLoopFunc::IterationEnd()+0x44) [0xffff9c5399c4]
libtile_fwk_interface.so(npu::tile_fwk::RecordLoopFunc::Iterator::operator!=(npu::tile_fwk::RecordLoopFunc::IteratorEnd const&)+0xfc) [0xffff9c539ea0]
```

## Possible Causes

This error occurs because the internal topological sorting of basic operators detects a cycle.

The root cause is that data is read from `a` and then written back to `a`. Since PyPTO describes a graph representation, `a` is treated as a whole during both read and write operations. As a result, the created connections form a cycle: `a → b → b.exp() → a`. PyPTO does not allow cycles in the constructed graph. The graph must be a DAG (Directed Acyclic Graph). This is why the error is reported.

![](../../figures/tensor/view_assemble_cycle_1.png)

## Solution

- Currently, the logic for reading and writing `a` needs to be defined in two separate graphs to avoid cycles within a single graph.
- After the SSA semantics of Assemble is launched in a future release, this usage will no longer cause issues.

![](../../figures/tensor/view_assemble_cycle_2.png)
