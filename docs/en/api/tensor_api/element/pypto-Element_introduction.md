# Introduction to pypto.Element

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-26T15:52:35.829Z pushedAt=2026-10-28T12:03:30.096Z nl=2 -->

In the PyPTO framework, the Element type is used to store scalar constants for constant expressions in computation operations. The built-in Python int type is typically mapped to the DT\_INT64 type, while the float type is mapped to the DT\_FP32 type. In scenarios where both Tensor and Scalar operand types are present, int and float are usually converted to the data type of the corresponding tensor.
