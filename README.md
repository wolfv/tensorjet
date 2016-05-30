# TensorFlow to C++ compiler

This library intends to compile a TensorFlow graph to C++.
However, it only supports super simple constructs currently (such as the binary operators +,-,* and /).

With your help we can create a very nice abstraction layer for hyper-fast numeric evaluation of tensorflow graphs.

This will, however, likely not speed up your TensorFlow models as this optimizes small graphs with few large matrix multiplications etc.
We are only hijacking TensorFlows nice graph abstraction for a numeric oriented library.