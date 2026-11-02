# F20000-F3FFFF

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:22:44.590Z pushedAt=2026-10-27T03:42:29.810Z nl=2 -->

## F21001 INVALID_OPERATION

**Error Description:**

Operation not allowed.

**Possible Causes:**

- An attempt was made to perform an operation that is not allowed.
- The tensor is written with different data for the second time.
- Incorrect operation context.

**Solution:**

- Check whether the operation is executed in the correct context.
- Ensure that the operation complies with system constraints.

## F21002 INVALID_TYPE

**Error Description:**

Invalid type.

**Possible Causes:**

- Input parameter type mismatch.
- A data type not supported by the API is used.

**Solution:**

- Check whether the type of the input parameter matches the requirement.
- Use a type supported by the API.

## F21003 INVALID_VAL

**Error Description:**

Invalid value.

**Possible Causes:**

- Parameter values (such as shape and offset dimensions) do not match and cannot be computed.
- The parameter value is invalid.

**Solution:**

- Check the parameter format.
- Use a valid parameter value.

## F21004 INVALID_PTR

**Error Description:**

Invalid pointer.

**Possible Causes:**

- The pointer is null.
- The pointer is not properly initialized.

**Solution:**

- Ensure that the pointer is properly initialized.
- Check the pointer validity before using it.

## F21005 OUT_OF_RANGE

**Error Description:**

Parameter out of range.

**Possible Causes:**

- The index is out of the valid range.
- The parameter value is out of the valid range.

**Solution:**

- Check the index range and use a valid index value.
- Use a parameter value within the valid range.

## F21006 IS_EXIST

**Error Description:**

The parameter/operation already exists.

**Possible Causes:**

- An attempt was made to create an object that already exists.
- Duplicate object name.

**Solution:**

- Check whether the object already exists.
- Use a unique object name.

## F21007 NOT_EXIST

**Error Description:**

The parameter/operation does not exist.

**Possible Causes:**

- A nonexistent object is accessed.
- The object is not properly registered.

**Solution:**

- Check whether the object exists.
- Ensure that the object is properly registered.

## F21008 DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED

**Error Description:**

A tensor with a dynamic shape cannot directly participate in computation.

**Possible Causes:**

- A tensor containing a `pypto.DYNAMIC` dimension is directly used as an operation operand.
- No static shape is sliced out via view in `pypto.loop` before computation.

**Solution:**

- Slice dynamic dimensions by a fixed view size.
- Express computation on the sliced static-shape tensor in `pypto.loop`.

## F21009 OP_DEPENDENCY_CYCLE

**Error Description:**

A cycle exists in the operator dependency graph, causing `GetSortedOperations` topological sorting to fail.

**Possible Causes:**

- Within the same JIT function, the **same tensor slot** is first **read** and then **written back** (for example, first read via `view`/`where`, then write the read result back to the same tensor via `assemble`), causing dependency edges to connect end-to-end and the graph to no longer be a DAG.

  ```python
  # Error example - read first, then write back to the same slot.
  tile = pypto.view(buf, ...)           # Read buf.
  pypto.assemble(tile, [i, 0], buf)     # Write tile back to buf -> forms a cycle.
  ```

- Renaming a Python variable does not change the slot. The framework treats a tensor as a whole node and does not distinguish partial slices.

**Solution:**

- **Separate read/write slots**: Read `buf_rd`, write `buf_wr`, and store the result in a third variable.
- **Pre-transform/Do not write back**: Apply transformations such as `where` before `assemble`; or write first and read later, where the write buffer is only used for `assemble` and the read buffer is not written back after reading.

  ```python
  # Correct example - write first, then read, and do not write back.
  tile = pypto.view(buf, ...)  # First split out tile.
  for i in pypto.loop(M, name="i_loop", idx_name="i"):
      pypto.assemble(tile, [i, 0], buf) # Only write buf.
  out = pypto.mul(buf, scale)           # Only read buf, write to a new tensor.
  ```

- **Graph splitting**: If read and write cannot be separated, split into two `@jit` functions.

## F29001 BAD_FD

**Error Description:**

Incorrect file descriptor status.

**Possible Causes:**

- Incorrect file descriptor status.
- The file is not properly opened or closed.

**Solution:**

- Check the file descriptor status.
- Ensure that files are properly opened and closed.

## F29002 INVALID_FILE

**Error Description:**

Invalid file content.

**Possible Causes:**

- The file content format is incorrect.
- The file content does not meet expectations.

**Solution:**

- Check the file content format.
- Use the correct file content.

## F3FFFF UNKNOWN

**Error Description:**

Unknown error.

**Possible Causes:**

- The error cause is unclassified/cannot be classified.

**Solution:**

- Visit the community and submit an [issue](https://gitcode.com/cann/pypto/issues).
