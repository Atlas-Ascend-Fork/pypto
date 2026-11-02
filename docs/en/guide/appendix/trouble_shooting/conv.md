# FC6XXX-FC8XXX

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:22:48.033Z pushedAt=2026-10-28T08:18:06.295Z nl=2 -->

## FC6101 OVER_BUFFER_LIMIT

**Error Description**

During the Operation stage, the TileShape configuration exceeds hardware cache space limits, including L0A, L0B, L0C, and L1 caches.

**Possible Causes**

- The L1, L0A, L0B, and L0C cache space exceeds the limit.

**Solution**

1. Convolution Forward reference: [Convolution Forward tileshape split constraint description](../../../api/tensor_api/config/pypto-set_conv_tile_shapes.md). Calculate whether the occupied space exceeds the limit.
2. Convolution Backward reference: [Convolution Backward tileshape split constraint description](../../../api/tensor_api/config/pypto-set_convbp_input_tile_shapes.md). Calculate whether the occupied space exceeds the limit.

## FC6201 EXPANDFUNC_TENSOR_OP_NULLPTR

**Error Description**

During the Tile graph split stage, when setting attributes for the MMAD (cube multiply-accumulate) operation, the input and output tensor graph node pointers are null.

**Possible Causes**

- The input/output tensors of the Conv Operation or Conv_backward_input Operation are not correctly passed in or are released prematurely, causing the tensor nodes to be missing during Tile graph expansion.

   ```python
   # Error example - In dynamic scenarios, the offset or shape passed to the view operation is abnormal, causing the fmap sub-tensor node to not be correctly constructed during Tile graph expansion.
   input_a_view = pypto.view(input_a, [tile_batch, 16, 64], [batch_offset, 0, 0])
   # If input_a itself is None or not correctly initialized, fmapTensorPtr is null during Tile graph expansion.
   out = pypto.conv(input_a_view, input_b, dtype, [1], [1, 1], [1], extend_params={}, groups=1)
   ```

- In dynamic shape scenarios, the sub-tensor chain constructed by view/assemble operations is abnormal, causing the tensor pointer passed to the MMAD node to be lost.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, and confirm whether the input/output tensor nodes of the Conv Operation or Conv_backward_input Operation are complete:

   ```python
   @pypto.frontend.jit(debug_options={"compile_debug_mode": 1})
   def conv_kernel():
   ```

2. In the output directory, use pto-toolkit to open the dumped compute graph, locate the Conv Operation or Conv_backward_input Operation node that reports the error, and check whether its input tensors and output tensors all exist and are non-null.
3. In dynamic shape scenarios, check whether the offset and shape parameters of `pypto.view` are valid, and confirm that the sub-tensor after the view operation can be correctly passed to `pypto.conv` or `pypto.conv_backward_input`.
4. If the above troubleshooting finds no issues, submit an issue or ticket via the CONV component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6202 EXPANDFUNC_TENSOR_ATTR_GET_FAILED

**Error Description**

Convolution Forward: In the Tile graph split stage, the original fmap/weight shape attributes (`CONV_ORI_FMAP_SHAPE_ATTR`/`CONV_ORI_WEIGHT_SHAPE_ATTR`) of the Conv Operation are not set. This check is triggered in NZ2NZ mode.
Convolution Backward: In the Tile graph split stage, the original gradOutput/weight shape attributes (`CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR`/`CONV_ORI_WEIGHT_SHAPE_ATTR`) of the Convbp Operation are not set. This check is triggered in NZ2NZ mode.

**Possible Causes**

- The Operation construction process does not correctly set the original shape attributes.

   ```txt
   # Error example - in NZ2NZ mode, the Conv Operation is missing the ori_fmap_shape attribute
   # Error log example:
   # Conv ori fmapshape should be set when InOut Tensor NZ mode.
   ```

- The Conv Operation or Conv_backward_input Operation is constructed in a non-standard way (for example, a custom Pass modified the Conv node attributes), causing the required attributes to be lost.

**Solution**

1. Confirm the runtime chip model and check whether the NZ2NZ path (`ConstructTensorGraphNZ2NZ`) is used.
2. Enable the compilation debug mode to dump the compute graph. Convolution Forward: search for the Conv Operation node in the dumped graph and check whether its attribute list contains `ori_fmap_shape` and `ori_weight_shape`; Convolution Backward: search for the Conv_backward_input Operation node in the dumped graph and check whether its attribute list contains `ori_gradOutput_shape` and `ori_weight_shape`.
3. Confirm whether the Operation is constructed through the standard `pypto.conv` or `pypto.conv_backward_input` API. If a custom Pass is used to modify the Conv node, check whether the shape attributes were accidentally deleted.
4. If the above troubleshooting finds no issues, submit an issue or ticket with the CONV component, and attach the complete error log, chip model information, and reproduction case.

## FC6203 EXPANDFUNC_TILE_OP_NULLPTR

**Error Description**

During the Tile graph split stage, a null pointer occurs in a newly generated node of the Tile graph, including: the current Function pointer is null, and the input/output tile tensor pointers are null.

**Possible Causes**

- The Conv Operation or Conv_backward_input Operation is not called inside a dynamic function decorated with `@pypto.frontend.jit`, causing failure to obtain the current Function pointer (`functionPtr` is null).

   ```python
   # Incorrect example - conv is not called inside a jit dynamic function, so functionPtr is null
   def not_under_jit_example(fmap, weight):
       output = pypto.conv(fmap, weight, dtype, [1, 1], [0, 0, 0, 0], [1, 1])
       return output  # functionPtr is null, triggering FC6203
   ```

- `hasBias=True` is configured but no bias tensor is passed, or the bias tensor is lost during Tile graph expansion, causing `biasTensorPtr` to be null.

   ```python
   # Incorrect example - hasBias is True but bias is passed abnormally in extend_params
   extend_params = {'bias_tensor': None}  # bias is None, hasBias is marked True but biasTensorPtr is null
   output = pypto.conv(fmap, weight, dtype, [1, 1], [0, 0, 0, 0], [1, 1], extend_params=extend_params)
   ```

- During Tile graph expansion, the construction of L0-level input/output sub-tensor nodes fails.

**Solution**

1. Confirm that the `pypto.conv` or `pypto.conv_backward_input` call is located inside a function decorated with `@pypto.frontend.jit`; it cannot be called directly in a normal Python function.
2. If using bias, confirm that `extend_params['bias_tensor']` is a valid tensor object, not None.
3. Enable the compilation debug mode to dump the compute graph, and check whether the tensor connection relationships of all nodes after Tile graph expansion are complete.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the complete error log and reproduction case.

## FC6204 EXPANDFUNC_PARAMS_INVALID

**Error Description**

In the Tile graph split stage, the number of input operands of the Conv Operation or Conv_backward_input Operation does not match the expected value. The expected number of operands = 2 (fmap + weight) + hasBias (0 or 1).

**Possible Causes**

- The operands of the Conv Operation or Conv_backward_input Operation are abnormally added or removed during graph propagation, for example, a custom Pass incorrectly adds or deletes input edges.

   ```txt
   # Error example - the number of operands of the Conv Operation is inconsistent with the hasBias flag
   # Error log example:
   # Operand vector size mismatch: Expected size: 3, actual size: 2, Conv Common Input: 2, hasBias: True
   ```

- Abnormal bias operand passing logic: The `hasBias` flag is **True** but the actual operands lack a bias, or `hasBias` is **False** but an extra bias exists in the operands.

**Solution**

1. Enable compilation debug mode to dump the compute graph, locate the failing Conv Operation or Conv_backward_input Operation node in the dumped graph, and check the number and type of its input edges.
2. Confirm that the number of input operands is consistent with the `hasBias` flag: the number of operands is 2 when there is no bias, and 3 when there is a bias.
3. If using a custom Pass, check whether the input edges of the Conv Operation or Conv_backward_input Operation are incorrectly modified.
4. If the above troubleshooting finds no issues, submit an issue or ticket via the CONV component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6205 EXPANDFUNC_INNER_STATUS_FAILED

**Error Description**

During the Tile graph partitioning stage, an internal function returns an abnormal value. This error code is a reserved error code.

**Possible Causes**

NA

**Solution**

1. This error code is a reserved code and is not used in the current version. If you encounter this error, submit an issue or ticket through the CONV component, and attach the complete error log and reproduction case.

## FC6301 CODEGEN_GET_ATTR_FAILED

**Error Description**

In the Codegen code generation stage, obtaining the CopyInMode or CopyOutMode attribute of the Conv TileOp or Conv_backward_input TileOp fails. These two attributes are set in the Tile graph expansion stage and read in the Codegen stage.

**Possible Causes**

- The CopyInMode/CopyOutMode attributes were not correctly set during the Tile graph expansion stage, usually caused by an abnormal Tile graph expansion process.

   ```txt
   # Error example: failure returned when reading the CopyInMode attribute in the Codegen stage
   # Error log example:
   # GenMemL1CopyInConv get CopyInMode failed.
   # GenMemL0CCopyOutConv get CopyOutMode failed.
   ```

- The Load/Store operation nodes of the Conv Operation or Conv_backward_input Operation are abnormally modified in the Pass stage, causing the attributes to be lost.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, locate the Load (CopyIn)/Store (CopyOut) operation nodes of Conv or Conv_backward_input in the dumped graph, and check whether their attributes contain `COPY_IN_MODE`/`COPY_OUT_MODE`.
2. Check whether a custom Pass is used to modify the Load/Store nodes of Conv or Conv_backward_input, and confirm that the attributes are not deleted by mistake.
3. In the generated kernel code file (`TENSOR***.cpp` in the `kernel_aicore` directory), search for the corresponding TileOp call and confirm whether the parameters are complete.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the Codegen component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6302 CODEGEN_CHECK_ATTR_INVALID

**Error Description**

During the Codegen code generation stage, the CopyInMode/CopyOutMode attribute value of the Conv TileOp or Conv_backward_input TileOp is not within the valid range, or the cutW attribute is 0.

**Possible Causes**

- The CopyInMode value is not within the range of `[ND2NZ, NZ2NZ, DN2NZ]`.

   ```txt
   # Error example - invalid CopyInMode/CopyOutMode attribute value or cutW is 0
   # Error log example:
   # GenMemL1CopyInConv CopyInMode is invalid: -1
   # GenMemL0CCopyOutConv CopyOutMode is invalid: 99
   # GenMemL0CCopyOutConv cutW should not be 0!
   ```

- The CopyOutMode value is not within the `{NZ2ND, NZ2NZ, NZ2DN}` range.
- The `cutW` attribute is **0**, causing the L0C-to-GM copy-out operation to fail to correctly split into blocks.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, locate the Conv or Conv_backward_input Load/Store operation nodes in the dumped graph, and check whether the `COPY_IN_MODE`/`COPY_OUT_MODE`/`CUT_W` attribute values are valid.
2. Valid values for `CopyInMode`: `ND2NZ` (1), `NZ2NZ` (2), `DN2NZ` (3); valid values for `CopyOutMode`: `NZ2ND` (0), `NZ2NZ` (1), `NZ2DN` (3).
3. Confirm that the Tile graph expansion process is not interfered with by custom passes. These attributes are automatically derived by the framework based on the input tensor format and chip model.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the Codegen component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6303 CODEGEN_CHECK_DIM_INVALID

**Error Description**

During the Codegen code generation stage, the src shape or offset dimension of the Conv or Conv_backward_input TileOp does not match the expected value. For 2D, the shape/offset should be 4-dimensional; for 3D, it should be 5-dimensional; and the valid shape of L0C should be 2-dimensional.

**Possible Causes**

- During the Tile graph expansion stage, the generated tensor shape dimensions are inconsistent with the convolution type. For example, the fmap shape of a 2D conv is incorrectly generated as 5-dimensional.

   ```txt
   # Error example - shape/offset dimension does not match the convolution type
   # Error log example:
   # GenMemL1CopyInConv shape should be 4-dim! (2D conv expects 4 dimensions, but the actual value is not 4-dimensional)
   # GenMemL1CopyInConv offset should be 4-dim! (2D conv expects 4 dimensions, but the actual value is not 4-dimensional)
   # GenMemL0CCopyOutConv valid shape should be 2-dim! (L0C expects 2 dimensions, but the actual value is not 2-dimensional)
   ```

- In dynamic shape scenarios, the dimension derivation of the valid shape is abnormal.
- The shape of the L0C tensor is not 2D (M, N), which may be caused by an L0C node construction error during Tile graph expansion.

**Solution**

1. Enable the compilation debug mode to dump the compute graph, locate the Conv or Conv_backward_input Load/Store node that reports the error in the dumped graph, and check the shape dimension of its src tensor.
2. Confirm that the shape dimensions match the convolution type: 1D conv corresponds to 3D (NCL), 2D conv corresponds to 4D (NCHW), and 3D conv corresponds to 5D (NCDHW).
3. If it is a dynamic shape scenario, check whether the sub-tensor shape dimensions constructed by `pypto.view` are consistent with the original input.
4. Check the generated kernel code file (in the `kernel_aicore` directory) and confirm that the `shape`/`offset` parameter dimensions of the TileOp call are correct.
5. If the above troubleshooting finds no issues, submit an issue or ticket through the Codegen component, and attach the complete error log, dumped compute graph, and reproduction case.

## FC6401 TILEOP_TENSOR_FORMAT_FAILED

**Error Description**

During the TileOp stage, the tensor hardware FORMAT verification fails. The Load operation of Conv or Conv_backward_input requires the src to be in GM format and the dst to be in L1 format. The Store operation requires the src to be in L0C format and the dst to be in GM format.

**Possible Causes**

- During the Tile graph expansion stage, an incorrect memory hierarchy is assigned to the tensor. For example, the dst of a Load operation is assigned to L0C instead of L1.

   ```txt
   # Error example - the src/dst hardware formats of the Load/Store operation do not match
   # Error log example (compile-time static_assert):
   # [TLoadConv Error]: Src format should be GM and Dst format should be L1
   # [TStoreConv Error]: Src format should be L0C and Dst format should be GM
   ```

- A custom pass modified the memory type attribute of the tensor, causing a src/dst format mismatch for Load/Store operations.

**Solution**

1. The verification corresponding to this error code is a compile-time `static_assert`, which reports an error during the kernel code compilation stage. In the generated kernel code file (under the `kernel_aicore` directory), search for `TLoadConv`/`TStoreConv` calls and check the tensor type declarations passed in.
2. Verify that the dst tensor of the Load operation is declared as L1 type (for example, `ConvTile<TileType::Mat, ..., Hardware::L1>`), and the src tensor is of GM type.
3. Verify that the src tensor of the Store operation is declared as L0C type, and the dst tensor is of GM type.
4. If the Tile graph expansion logic has not been modified but this error still occurs, submit an issue or ticket through the CONV component, with the kernel code file and reproduction case attached.

## FC6402 TILEOP_SHAPE_SIZE_FAILED

**Error Description**

In the TileOp stage, the shape size validation of the L0C tensor fails. The Store operation requires the L0C shape to be 2-dimensional (M, N).

**Possible Causes**

- In the Tile graph expansion stage, a non-2D shape is constructed for the L0C tensor, which may be caused by an anomaly in the output shape derivation of the Conv MMAD node.

   ```txt
   # Error example - L0C tensor shape is not 2-dimensional
   # Error log example (compile-time static_assert):
   # L0C shape size should be 2 Dim
   ```

- A custom pass modified the shape definition of the L0C tensor.

**Solution**

1. The verification corresponding to this error code is a compile-time `static_assert`. In the generated kernel code file, search for the `TStoreConv` call and check the shape declaration of the L0C tensor.
2. Confirm that the L0C tensor shape is 2-dimensional, i.e., `(M, N)`.
3. Enable the compilation debug mode to dump the compute graph, and check the shape of the L0C tensor node from the MMAD output to the Store operation in the dumped graph.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the kernel code file and reproduction case.

## FC6403 TILEOP_STC_SHAPE_INVALID

**Error Description**

In the TileOp stage, the static shape (compile-time constant shape) validation is invalid. Some dimensions of the Conv or Conv_backward_input TileOp are required to be compile-time constants and cannot be modified at runtime.

**Possible Causes**

- Some dimensions in the TileShape configuration must be compile-time constants but are set to dynamic values, causing `std::tuple_element` to fail during compilation.

   ```txt
   # Error example: The static dimension configuration of TileShape is abnormal, for example, tileCinFmap or tileN is set to 0, causing bufferSize = 0
   ```

- The buffer size calculation result of TileShape is **0** or negative (for example, a dimension is configured as **0**).

**Solution**

1. Check whether all dimension values of TileShape set by `pypto.set_conv_tile_shapes` or `pypto.set_convbp_tile_shapes` are positive integers.
2. Confirm that the dimensions in TileShape that need to be compile-time constants (such as `tileN` and `tileWout`) are correctly declared using `pypto.symbolic_scalar` in dynamic shape scenarios.
3. Enable the compilation debug mode and check whether the TileShape expansion results in the compilation log are valid.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the complete error log and reproduction case.

## FC6404 TILEOP_INDEX_INVALID

**Error Description**

During the TileOp stage, the index validation for shape/stride access failed. The shape/stride access index must be less than 5.

**Possible Causes**

- During the Tile graph expansion stage, an attempt was made to access the fifth dimension or higher (index >= 5) of a tensor shape/stride. This may be caused by an anomaly in tensor shape dimension derivation.

   ```txt
   # Error example - shape/stride access index >= 5
   # Error log example (compile-time static_assert):
   # Idx should be less than 5
   ```

- The tensor of a 2D conv is incorrectly processed as a 3D conv, or vice versa, causing an out-of-bounds dimension access.

**Solution**

1. The corresponding verification for this error code is a compile-time `static_assert`. Check the TileOp call corresponding to the error, and verify whether the passed index parameter is valid (must be < 5).
2. Enable the compilation debug mode to dump the compute graph, and check whether the tensor shape dimensions of the Conv or Conv_backward_input Load/Store nodes match the convolution type (2D convolution is 4 dimensions + pad = 5 dimensions, and 3D convolution is 5 dimensions + pad = 6 dimensions for internal processing).
3. Confirm that the convolution type is determined correctly: 1D convolution input is 3 dimensions, 2D convolution input is 4 dimensions, and 3D convolution input is 5 dimensions.
4. If the above troubleshooting finds no issues, submit an issue or ticket through the CONV component, and attach the kernel code file and reproduction case.
