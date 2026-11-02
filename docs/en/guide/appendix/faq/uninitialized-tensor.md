# Using an Uninitialized Tensor

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:20:46.305Z pushedAt=2026-10-27T02:45:18.699Z nl=2 -->

## Symptoms

You declared a tensor using `pypto.tensor` and mistakenly assumed that, similar to `torch.empty`, it would allocate a block of uninitialized memory. Reading the tensor directly (for example, using `view`) before writing to it again causes a framework verification error or precision issue.

## Possible Causes

In PyPTO, in addition to using `pypto.full`, `pypto.zeros`, and other tensor declarations that explicitly include initialization behavior, you can also use `pypto.tensor` to declare a tensor, but this API does not include initialization behavior. In PyPTO, every tensor must be written before being read, meaning there must be a producer before a consumer. An uninitialized tensor does not allocate memory. The framework typically checks for direct use of a no-producer tensor and reports an error, but sometimes, due to a missed check, this may lead to on-device precision errors.

## Solution

Avoid using uninitialized tensors.
