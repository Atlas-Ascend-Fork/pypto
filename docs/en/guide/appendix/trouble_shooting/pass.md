# F4XXXX-F5XXXX

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:27:08.315Z pushedAt=2026-10-27T03:50:22.765Z nl=2 -->

## F40000 TENSOR_NULL_POINTER

**Error Description**

A null pointer reference exists in the tensor or its associated operations.

**Possible Causes**

- The producer of the tensor is null.
- The consumer of the tensor is null.
- The input tensor of the Operation is null.
- The output tensor of the Operation is null.
- A null consumer exists among the consumers of the tensor.
- A null producer exists among the producers of the tensor.

**Solution**

1. Check whether the producer and consumer of the reported tensor are correctly created.
2. Check whether the input and output tensors of the Operation are null.
3. Check whether any connection between the tensor and Operation is missing during graph construction.

## F40001 TENSOR_INVALID_MEMORY_TYPE

**Error Description**

The memory type of the tensor is configured with an invalid or mismatched value.

**Possible Causes**

- The memory type of the tensor is an invalid or undefined value.
- The memory type of the tensor is incompatible with the requirements of the subgraph or compute unit where it resides.
- A dynamic shape tensor uses an invalid memory type configuration.
- The boundary tensor does not use the specified memory type.

**Solution**

1. Check whether the memory type of the tensor is a valid value supported by the framework.
2. Check whether the subgraph, compute unit, and memory type of the tensor match.
3. If the tensor is used across subgraphs, check whether the memory type configuration of the boundary tensor meets the requirements.

## F40002 TENSOR_SUBGRAPH_BOUNDARY

**Error Description**

The tensor used across subgraphs is not correctly marked as a boundary.

**Possible Causes**

- The DDR tensor is not marked as a subgraph boundary.
- Cross-subgraph tensors are not marked as subgraph boundary.
- The subgraph ID of the tensor is `NOT_IN_SUBGRAPH`.

**Solution**

1. Check whether the cross-subgraph tensor has the subgraph boundary flag set.
2. Check whether the boundary attribute of the DDR boundary tensor is complete.
3. Check whether the subgraph ID of the tensor has been correctly assigned to a valid subgraph.

## F40003 TENSOR_SHAPE_MISMATCH

**Error Description**

The shape or memory type of the tensor does not meet the input/output constraints of the Operation.

**Possible Causes**

- The input/output tensor shape of the Operation does not meet the constraints.
- The input/output tensor memory type of the Operation does not meet constraints.

**Solution**

1. Check whether the input and output tensor shapes meet the semantic requirements of the reported Operation.
2. Check whether the input and output dimensions of view-type Operations such as reshape, view, and assemble are consistent or derivable.
3. Check whether the tensor memory type matches the input/output paths supported by the Operation.

## F40004 TENSOR_UNSUPPORTED_DATATYPE

**Error Description**

The data type of the tensor is not supported by the Operation.

**Possible Causes**

- The Operation does not support the dtype of the input tensor.
- The Operation does not support the dtype of the output tensor.
- The input/output tensor dtype combination does not meet the Operation constraints.

**Solution**

1. Check the data type range supported by the reported Operation.
2. Adjust the input or output tensor dtype to a type supported by the Operation.
3. If type conversion is required, explicitly insert valid cast or conversion logic before entering the Operation.

## F40005 TENSOR_MEMORY_ALLOCATION

**Error Description**

The memory size, address range, or alignment configuration of the tensor does not meet the memory allocation constraints.

**Possible Causes**

- The same memory region is illegally overlapped by multiple tensors.
- Improper memory segment division causes address out-of-bounds.
- The tensor memory size is 0 or exceeds the valid allocation range.
- Dynamic memory allocation attributes are missing or illegally configured.
- The tensor memory alignment does not comply with hardware constraints.

**Solution**

1. Check whether the memory address, offset, and size of the tensor have out-of-bounds or illegal overlap.
2. Check whether the dynamic memory-related attributes are complete.
3. Check whether the tensor memory size and alignment meet the hardware and framework constraints.

## F40006 TENSOR_DYNAMIC_ATTR

**Error Description**

The dynamic attribute of the Operation or the `dynValidShape` of the tensor is missing or incorrectly configured.

**Possible Causes**

- The dynamic attribute of the Operation is missing.
- The `dynValidShape` of the tensor is empty.

**Solution**

1. Check whether the `dynValidShape` of the tensor has been correctly set in dynamic shape scenarios.
2. Check whether the related Operation has the input information required for dynamic attribute inference.
3. If the pass modifies the view-type Operation or tensor shape, check whether `dynValidShape` is updated synchronously.

## F41000 OP_INVALID_OPERAND_COUNT

**Error Description**

The number of inputs, outputs, or sideband inputs of the Operation does not comply with the Operation constraints.

**Possible Causes**

- The actual number of input tensors of the Operation is non-compliant.
- The actual number of output tensors of the Operation does not comply with the specification.
- The number of control dependencies or sideband inputs does not comply with the constraint.

**Solution**

1. Check whether the number of input tensors is correct based on the Operation semantics.
2. Check whether the number of output tensors is correct based on the Operation semantics.
3. If the Operation uses control dependencies or sideband inputs, check whether the number and order of the corresponding inputs comply with the constraints.

## F41001 OP_NULL_POINTER

**Error Description**

A null pointer reference exists in the Operation, Operation attribute, or input/output Tensor list.

**Possible Causes**

- The Operation is null.
- The op attribute of the Operation is null.
- The IOperands or OOperands of the Operation are null.

**Solution**

1. Check whether the Operation is successfully created and added to the function.
2. Check whether the op attribute is set for the Operation that requires attributes.
3. Check whether the input/output tensor list of the Operation is a null pointer.

## F41002 OP_INVALID_OPCODE

**Error Description**

The opcode of the Operation does not belong to a valid type supported by the current function, subgraph, or pass stage.

**Possible Causes**

- The Operation type is invalid.
- The current pass or current graph structure does not support this opcode.

**Solution**

1. Check whether the opcode of the reported Operation is a valid type supported by the framework.
2. Check whether the opcode is allowed to appear in the current function, subgraph, or pass phase.
3. If it is a newly added Operation, check whether the corresponding pass has implemented the support logic.

## F41003 OP_PRODUCER_CONSUMER

**Error Description**

The Operation lacks a valid producer or consumer, or the bidirectional tensor connection relationships are inconsistent.

**Possible Causes**

- The Operation has no producer.
- The Operation has no consumer.
- The producer or consumer relationship between the Operation and the input/output tensor is inconsistent.

**Solution**

1. Check whether the input tensor of the Operation has a valid producer.
2. Check whether the output tensor of the Operation has a valid consumer.
3. Check whether the bidirectional connection relationships of the tensor are consistent, that is, the Operation references the tensor, and the tensor also records the corresponding producer or consumer.

## F41004 OP_SPECIAL_CONSTRAINT

**Error Description**

The connection relationship or target memType of the Operation does not meet the special constraints of the current Operation.

**Possible Causes**

- The producer or consumer Operation type of the Operation is non-compliant.
- The to memType type of the Operation is non-compliant.

**Solution**

1. Based on the reported Operation, check whether its producer and consumer types meet the special constraints.
2. Check whether the target memory type of the Operation falls within the allowed range.
3. For special Operations such as reshape, view, assemble, and copy, check whether the preceding and following connections comply with the pass constraints.

## F41005 OP_NESTING_DEPTH

**Error Description**

The Operation nesting depth exceeds the framework limit.

**Possible Causes**

- The Operation nesting depth exceeds the framework limit.

**Solution**

1. Check whether the nesting structure where the reported Operation resides is too deep.
2. Simplify the nesting depth, or split them into multiple valid intermediate steps.
3. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## F41006 OP_SEQUENCE_ERROR

**Error Description**

The Operation sequence contains an operation combination that is not supported by the current pass or backend execution.

**Possible Causes**

- Disallowed Operations exist.
- An unsupported combination of Operations exists.
- The Operation sequence does not comply with the current pass or backend execution constraints.

**Solution**

1. Check whether the combination of adjacent Operations is valid based on the error location.
2. Adjust the Operation sequence or split the unsupported combination.
3. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## F42000 FUNCTION_GRAPH_STRUCTURE

**Error Description**

The incast, outcast, Operation, or subgraph topology of the function is incomplete or invalid.

**Possible Causes**

- The function contains a null operation.
- The incast of the function is empty.
- The outcast of the function is empty.
- Circular dependencies exist in the function.
- The subgraph topology is incorrect.
- The subgraph ID is out of range.
- An empty subgraph exists.

**Solution**

1. Check whether the function contains valid incast and outcast.
2. Check whether Operations in the function contain null pointers.
3. Check whether the function has circular dependencies or invalid subgraph structures.
4. Check whether the subgraph ID is within the valid range and ensure that no empty subgraph exists.

## F42001 FUNCTION_BOUNDARY_COMPLETENESS

**Error Description**

The incast, outcast, or Operation subgraph ownership of the function is incomplete.

**Possible Causes**

- The incast has no consumer.
- The outcast has no producer.
- The subgraph ID of the Operation is negative and the Operation is not a NOP.

**Solution**

1. Check whether all incasts are connected to at least one consumer.
2. Check whether all outcasts have a valid producer.
3. Check whether the subgraph ID of the Operation is valid; Operations other than NOP should not use an invalid negative subgraph ID.

## F42002 FUNCTION_GRAPH_CONNECTION

**Error Description**

The input graph, output graph, subgraph boundary, or graph edge index relationship of the function is inconsistent.

**Possible Causes**

- The input graph and output graph do not match.
- The subgraph boundary tensor is not correctly marked.
- The edge index exceeds the Operation list range.
- The magic number of the operation cannot be found.

**Solution**

1. Check whether the function input graph and output graph are consistent.
2. Check whether the cross-subgraph tensor has the boundary marker correctly set.
3. Check whether the graph edge index is out of bounds.
4. Check whether the Operation magic number is unique and can be found in the function.

## F42003 FUNCTION_EXPAND_FEATURE

**Error Description**

The function expansion state or the temporary tensor connection relationship after expansion is incorrect.

**Possible Causes**

- The **ExpandFunctionAccelerate** flag is not reset to **false**.
- A locally defined temporary tensor is used as an operation input but has no producer.

**Solution**

1. Check whether the function expand-related flags are restored to the expected state after expansion.
2. Check whether a local temporary tensor is directly used as an Operation input.
3. If the temporary tensor needs to participate in computation, ensure that it is produced by a legitimate Operation.

## F42004 FUNCTION_MEMORY_REACHABILITY

**Error Description**

The memory type conversion in the function is unreachable.

**Possible Causes**

- The input/output memory type of the Operation is unreachable.
- The input/output memory type conversion path does not exist.

**Solution**

1. Check whether a valid conversion path exists for the Operation input/output memory type.
2. Check whether the configuration declares the reachability relationship for the corresponding memory type.
3. If the conversion path is missing, adjust the tensor memory type or add a valid conversion Operation.

## F42005 FUNCTION_UNIQUENESS

**Error Description**

The magic number of the Operation or tensor in the function is duplicated.

**Possible Causes**

- The magic number of the Operation is duplicated.
- The magic number of the tensor is duplicated.

**Solution**

1. Check whether the Operation magic number is unique within the function.
2. Check whether the tensor magic number is unique within the function.
3. If the duplicate identifier is introduced by a node added by a pass, check the creation method of the new Operation or tensor.

## F42006 FUNCTION_SPECIAL_STRUCTURE

**Error Description**

The function contains a structure that does not comply with topology, subgraph nesting, or special operator combination constraints.

**Possible Causes**

- There is a special node connection mode that does not comply with topology specifications.
- The subgraph nesting structure does not comply with framework constraints.
- The function contains a special operator combination structure that is not allowed.

**Solution**

1. Check the producers, consumers, and topological order of the special nodes.
2. Check whether the subgraph nesting structure complies with framework constraints.
3. Split or adjust unsupported special operator combinations.

## F43000 GRAPH_LOOP_DETECTION

**Error Description**

A circular dependency that cannot be topologically sorted exists in the graph or function.

**Possible Causes**

- OperationLoopCheck fails. A circular dependency exists.
- LoopCheck failed. A loop exists.

**Solution**

1. Check whether a circular dependency exists in the reported function or graph.
2. Locate the Operations on the loop based on the error log.
3. Modify the computation logic or connection relationships to eliminate the data loop.

## F43001 GRAPH_TOPOLOGY_STRUCTURE

**Error Description**

The subgraph topology, parent-child graph ID relationship, or edge index of the graph does not comply with topology constraints.

**Possible Causes**

- The subgraph topology is incorrect.
- The parent-child subgraph ID relationship is incorrect. The parent subGraphId must be less than or equal to the subGraphId.
- The edge index exceeds the Operation list range.

**Solution**

1. Check whether the subgraph topology order meets the requirement that the producer precedes the consumer.
2. Check whether the parent-child subgraph ID relationship complies with the constraint.
3. Check whether the graph edge index is within the valid range of the Operation list.

## F43002 GRAPH_SUBGRAPH_EMPTY

**Error Description**

An empty subgraph exists in the graph partitioning result.

**Possible Causes**

- The subgraph is empty.
- Graph partitioning or pass node deletion leaves an empty subgraph.

**Solution**

1. Check whether the subgraph partitioning result contains an empty subgraph.
2. Check whether deletion or elimination passes synchronize subgraph information cleanup.
3. Adjust the graph structure or pass logic to prevent generating empty subgraphs.

## F43003 GRAPH_SUBGRAPH_ID_INVALID

**Error Description**

The subgraph ID of the Operation is negative or exceeds the subgraph count range.

**Possible Causes**

- The subgraph ID is negative and the operation is not a NOP operation.
- The subgraph ID exceeds the range of totalSubGraphNum.

**Solution**

1. Check whether the subgraph ID of the Operation is within the valid range.
2. Check whether totalSubGraphNum is consistent with the actual number of subgraphs.
3. For Operations other than NOP, avoid using an invalid negative subgraph ID.

## F43004 GRAPH_EDGE_CONSISTENCY

**Error Description**

The incoming edges, outgoing edges, or node index relationships of the graph are inconsistent.

**Possible Causes**

- The incoming edge graph and outgoing edge graph have mismatched sizes.
- The position of the node in the input graph exceeds the range of the output graph.
- The node exists in the input graph but is not found in the output graph.
- There are untraversed edges in the outgoing edge graph.

**Solution**

1. Check whether the incoming edge graph and outgoing edge graph are updated synchronously.
2. Check whether the node index is within the valid range of the graph structure.
3. Check whether edge relationships are synchronously maintained after adding, deleting, or replacing an Operation.

## F43005 GRAPH_COLOR_CONSISTENCY

**Error Description**

The graph coloring input/output, subgraph mapping, or coloring edge relationships are inconsistent.

**Possible Causes**

- The consistency check between the input colored graph and the output colored graph failed.
- The output colored graph fails to match the input.
- Edges between the original Operation and the subgraph Operation are missing in the output colored graph.
- Edges in the output colored graph have no corresponding edges in the output graph.

**Solution**

1. Check whether the graph coloring input and output information is consistent.
2. Check whether the mapping between the original operations and the subgraph operations is complete.
3. Check whether all edges in the output colored graph can be found in the output graph.

## F43006 GRAPH_READY_STATE

**Error Description**

The readyState in graph topology traversal is inconsistent with the predecessor count.

**Possible Causes**

- Inconsistent ready state in the topology.
- readyState does not match the negative predecessor count.

**Solution**

1. Check whether the node readyState is correctly updated during topology traversal.
2. Check whether the node predecessor count matches the actual number of input edges.
3. Check whether the ready state is recomputed after graph construction or modification.

## F43007 GRAPH_AIV_AIC_MIX

**Error Description**

Incompatible compute units or memory types are mixed in the same subgraph.

**Possible Causes**

- Both AIV and AIC operations exist in the subgraph.
- The subgraph contains both UB and L0/L1 memory type tensors.

**Solution**

1. Check whether AIV and AIC operations are mixed in the same subgraph.
2. Check whether UB and L0/L1 memory type tensors are mixed in the same subgraph.
3. Re-partition the subgraph based on the compute unit and memory type constraints.

## F44000 CONFIG_MEMORY_TYPE_REACHABLE

**Error Description**

The configuration lacks a reachable conversion path between the input/output memory types.

**Possible Causes**

- The input/output memory types are unreachable.
- The memory type conversion path does not exist.

**Solution**

1. Check whether there is a reachable path for the input/output memory type in the configuration.
2. Check whether the tensor memory type is incorrectly configured.
3. Adjust the memory type configuration or insert a valid memory type conversion path.

## F44001 CONFIG_SUBGRAPH_BOUNDARY

**Error Description**

The boundary marking of a cross-subgraph tensor is missing.

**Possible Causes**

- The DDR tensor is not marked as a subgraph boundary.
- Cross-subgraph tensors are not marked as subgraph boundaries.

**Solution**

1. Check whether DDR tensors are correctly marked as subgraph boundaries.
2. Check whether cross-subgraph tensors have the **boundary** attribute set.
3. Check whether boundary tensor information is synchronized after subgraph partitioning.

## F44002 CONFIG_TENSOR_MEMORY_TYPE

**Error Description**

The tensor memory type in the configuration does not match the input/output requirements of the subgraph or Operation.

**Possible Causes**

- Memory type mismatch.

**Solution**

1. Check whether the tensor memory type is a valid value supported by the framework.
2. Check whether the tensor memory type matches the requirements of the subgraph it belongs to and the Operation input/output.
3. Adjust the tensor memory type or configuration items based on the error context.

## F44003 CONFIG_FILE_FAILED

**Error Description**

Failed to read, parse, or load the configuration file.

**Possible Causes**

- Failed to open the configuration file.
- Configuration file read failure.
- The corresponding configuration item does not exist in the configuration file.
- The specified configuration key does not exist under the corresponding tab in the configuration file.
- Configuration item read failure. The specified configuration information does not exist.

**Solution**

1. Check whether the configuration file path is correct and whether the file exists and is readable.
2. Check whether the configuration file content format is correct.
3. Check whether the reported configuration item, tab, and key exist.
4. If the configuration file is generated by a tool, regenerate the configuration file and retry.
