# Introduction to pypto.Element

In the PyPTO framework, the Element type is used to store scalar constants for constant expressions in computation operations. The built-in int type in Python is usually mapped to the DT\_INT64 type, and the float type is mapped to the DT\_FP32 type. When both Tensor and Scalar operand types are present, int and float are usually converted to the data type of the corresponding tensor.
