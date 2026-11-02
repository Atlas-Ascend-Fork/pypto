# FC3XXX-FC5XXX

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:26:41.327Z pushedAt=2026-10-28T08:18:32.765Z nl=2 -->

## FC3000 ERR_PARAM_INVALID

**Error Description**

Invalid Matmul internal input parameter: the values of parameters such as **Shape** and **Format** within the framework do not meet the constraints.

**Possible Causes**

NA

**Solution**

1. Check the [pypto.matmul](../../../api/tensor_api/operation/pypto-matmul.md) and [pypto.scaled_mm](../../../api/tensor_api/operation/pypto-scaled_mm.md) documents to confirm that the inputs and outputs meet the requirements.
2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC3001 ERR_PARAM_MISMATCH

**Error Description**

Matmul internal input parameter mismatch: Inconsistencies in dimensions or other attributes between framework-internal parameters.

**Possible Causes**

NA

**Solution**

1. Check the [pypto.matmul](../../../api/tensor_api/operation/pypto-matmul.md) and [pypto.scaled_mm](../../../api/tensor_api/operation/pypto-scaled_mm.md) documents to confirm that the input/output meets the requirements.
2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC3002 ERR_PARAM_UNSUPPORTED

**Error Description**

Matmul internal input parameter is not supported: The framework internally uses an unsupported parameter combination.

**Possible Causes**

- The `scale_tensor` data type constraint is violated: `scale_tensor` is neither `DT_UINT64` nor `DT_INT64`; or the input/output data types in quantization/dequantization scenarios are invalid (e.g., `DT_INT8` input with `DT_FP16` output, or any input/output being `DT_INT8`).

**Solution**

1. Check the [pypto.matmul](../../../api/tensor_api/operation/pypto-matmul.md) and [pypto.scaled_mm](../../../api/tensor_api/operation/pypto-scaled_mm.md) documents to confirm that the input/output meets the requirements.
2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC5000 ERR_RUNTIME_NULLPTR

**Error Description**

Matmul runtime error: An empty tensor occurred during Matmul runtime.

**Possible Causes**

NA

**Solution**

1. Verify that the input and output tensors passed to the Matmul API are non-null and have completed address allocation, and check for any nullptr.

   ```python
   # Correct example - input Tensors have all been allocated with data
   a = pypto.tensor([16, 32], pypto.DT_FP16, "a")
   b = pypto.tensor([32, 64], pypto.DT_FP16, "b")
   out = pypto.matmul(a, b, pypto.DT_FP16)
   ```

2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## FC5002 ERR_RUNTIME_LOGIC

**Error Description**

Matmul runtime logic exception: The pre-check failed or the computation flow entered an exception branch.

**Possible Causes**

- The user input failed the validity check.
- An empty tensor appeared during the internal execution of Matmul.

**Solution**

1. Check the [pypto.matmul](../../../api/tensor_api/operation/pypto-matmul.md) and [pypto.scaled_mm](../../../api/tensor_api/operation/pypto-scaled_mm.md) documents to confirm that the input/output meets the requirements.
2. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).
