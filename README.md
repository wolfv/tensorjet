# TensorFlow to C++ compiler

This library intends to compile a TensorFlow graph to C++.
However, it only supports super simple constructs currently (such as the binary operators +,-,* and /).

With your help we can create a very nice abstraction layer for hyper-fast numeric evaluation of tensorflow graphs.

This will, however, likely not speed up your TensorFlow models as this optimizes small graphs with few large matrix multiplications etc.
We are only hijacking TensorFlows nice graph abstraction for a numeric oriented library.

## How to this?

First, run `sudo pip install cppimport`. Or actually, just grab cppimport from here: https://github.com/tbenthompson/cppimport and patch it by removing "-Wall" and "-Werror" because I cannot compile with those two flags.

Then generate some sweet C++ from the test.py: `python test.py > crazy_stuff.cpp`.

Now use the black magic of cppimport to import your model:

```python
import cppimport
import crazy_stuff

t = crazy_stuff.TensorJet()
a = t.run(100)
assert(a[0] == 100**4)
```

### Congrats