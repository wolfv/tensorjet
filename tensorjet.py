import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.types_pb2 import DataType
import IPython
import subprocess

class_template = """
#include <cmath>
#include <tuple>
#include <pybind11/pybind11.h>


{consts}
class TensorJet {{
public:
    std::tuple<{ret_types}> fn({args}) {{
        {ops}
        return std::make_tuple({ret_names});
    }}
private:
    {states}
}};

namespace py = pybind11;

void pyexport(py::module& m) {{
    py::class_<TensorJet>(m, "TensorJet")
        .def(py::init<>())
        .def("run", &TensorJet::fn)
    ;
}}
"""

sess = tf.Session()
graph = None

def beauty(text):
    p = subprocess.Popen(['clang-format'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE)
    stdout, stderr = p.communicate(input=text)
    return stdout

class Placeholder():
    fmt = '{type} {name}'
    def __init__(self, el):
        self.name = sanitize_name(el.name)
        self.el = el
        try:
            self.shape = [int(dim.size) for dim in el.attr['shape'].shape.dim]
        except:
            self.shape = []

    def __repr__(self):
        if self.shape:
            tmpl = '<{}>'.format(', '.join(str(s) for s in self.shape))
            return self.fmt.format(type='Eigen::Matrix' + tmpl, name=self.name)
        else:
            return self.fmt.format(type='float', name=self.name)


class OpRegister(object):
    def __init__(self, el):
        self.el = el


class PowOp(OpRegister):
    mat_pow = 'auto {name} = {mat}.pow({exp});'
    scalar_pow = 'auto {name} = std::pow({inp1}, {exp});'
    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.input1 = sanitize_name(self.el.inputs[0].name)
        self.input2 = sanitize_name(self.el.inputs[1].name)

    def __repr__(self):
        return self.scalar_pow.format(name=self.name, inp1=self.input1, exp=self.input2)

    @classmethod
    def registers(cls, op):
        return op == 'Pow'


class BinaryOp(OpRegister):
    op_map = {
        'Add': '+',
        'Sub': '-',
        'Mul': '*',
        'Div': '/',
    }
    fmt = 'auto {name} = {in1} {op} {in2};'

    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.input1 = sanitize_name(self.el.inputs[0].name)
        self.input2 = sanitize_name(self.el.inputs[1].name)

    def __repr__(self):
        return self.fmt.format(name=self.name, 
                               in1=self.input1, 
                               in2=self.input2,
                               op=self.op_map[self.el.type])

    @classmethod
    def registers(cls, op):
        return op in cls.op_map.keys()


class UnaryOp(OpRegister):
    op_map = {
        'Neg': '-',
    }
    fmt = '{name} = {op}{in1};'

    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.input1 = sanitize_name(self.el.inputs[0].name)

    def __repr__(self):
        return self.fmt.format(name=self.name, in1=self.input1,
                               op=self.op_map[self.el.type])

    @classmethod
    def registers(cls, op):
        return op in cls.op_map.keys()

class AssignOp(OpRegister):
    fmt = '{in1} = {in2};'
    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.input1 = sanitize_name(self.el.inputs[0].name)
        self.input2 = sanitize_name(self.el.inputs[1].name)

    def __repr__(self):
        return self.fmt.format(in1=self.input1, in2=self.input2)
    @classmethod
    def registers(cls, op):
        return op == 'Assign'


class IdentityOp(OpRegister):
    fmt = 'auto {name} = {in1};'
    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.input1 = sanitize_name(self.el.inputs[0].name)

    def __repr__(self):
        return self.fmt.format(name=self.name, in1=self.input1)
    @classmethod
    def registers(cls, op):
        return op == 'Identity'

class SqueezeOp(OpRegister):
    fmt = 'auto {name} = {in1};'
    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.input1 = sanitize_name(self.el.inputs[0].name)

    def __repr__(self):
        return self.fmt.format(name=self.name, in1=self.input1)

    @classmethod
    def registers(cls, op):
        return op == 'Squeeze'

class SliceOp(OpRegister):
    fmt = 'auto {name} = {in1}[view{view}];'
    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.input1 = sanitize_name(self.el.inputs[0].name)
        self.input2 = sanitize_name(self.el.inputs[1].name)
        self.input3 = sanitize_name(self.el.inputs[2].name)
        self.is1 = self.el.inputs[1].eval()
        self.is2 = self.el.inputs[2].eval()
        zip_slice = zip(self.is1, self.is2)
        self.view = ''.join(['({}, {})'.format(a[0], a[1]) for a in zip_slice])
    def __repr__(self):
        return self.fmt.format(name=self.name, 
            in1=self.input1,
            view=self.view)

    @classmethod
    def registers(cls, op):
        return op == 'Slice'


def Op(el):
    for cls in OpRegister.__subclasses__():
        if cls.registers(el.type):
            return cls(el)

dtype_map = {
    tf.float32: 'float',
    tf.float64: 'double',
    tf.int32: 'int',
    tf.float32_ref: 'float',
    tf.float64_ref: 'double',
    tf.int32_ref: 'int',
}

def sanitize_name(name):
    return name.split(':')[0].replace('/', '___')

def get_type(el):
    shape = el.get_shape()
    try:
        dtype = el.dtype
    except:
        dtype = 'float'
    if not shape or len(shape) == 0 or len(shape) == 1 and shape[0] == 1:
        return 'float'
    else:
        tmpl = '<{}>'.format(', '.join(str(s) for s in 
            [dtype_map[dtype]] + [s for s in shape]))
        return '{type}'.format(type='Array' + tmpl)


class Constant:
    fmt = 'constexpr {dtype} {name} = {val};'

    def __init__(self, el):
        self.el = el
        self.name = sanitize_name(el.name)
        self.value = el.values()[0]
        self.dtype = get_type(self.value)

    def get_value(self):
        return self.value.eval(session=sess)

    def __repr__(self):
        return self.fmt.format(
            dtype=self.dtype, 
            name=self.name,
            val=self.get_value())

class Variable:
    def __init__(self, var):
        self.var = var
        self.value = var.values()[0]
        self.dtype = self.value.dtype
        self.shape = self.value.get_shape()
        self.name = sanitize_name(var.name)

    def __repr__(self):
        return '{type} {name};'.format(type=get_type(self.value),
                                       name=self.name)


class ClassBuilder:

    def extract_return(self, out):
        self.return_names = [r.name.split(':')[0] for r in out]
        self.return_types = [get_type(r) for r in out]

    def __init__(self, out, g):
        global graph
        graph = g
        # find placeholder
        # self.proto = proto
        self.graph = g
        self.args = []
        self.ops = []
        self.states = []
        self.constants = []
        self.extract_return(out)

        for el in g.get_operations():
            if el.type == 'Placeholder':
                self.args.append(el)
            elif el.type == 'Variable':
                self.states.append(el)
            elif el.type == 'Const':
                self.constants.append(el)
            else:
                self.ops.append(el)

    def build(self):
        ret = class_template.format(
            consts='\n'.join([repr(Constant(const)) for const in self.constants]),
            args=', '.join([repr(Placeholder(p)) for p in self.args]),
            states='\n'.join([repr(Variable(v)) for v in self.states]),
            ops='\n'.join([repr(Op(op)) for op in self.ops]),
            ret_types=', '.join(self.return_types),
            ret_names=', '.join(self.return_names)
        )
        print(beauty(ret))