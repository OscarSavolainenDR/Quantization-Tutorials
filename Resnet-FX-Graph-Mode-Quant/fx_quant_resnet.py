import torch
import torch.quantization as tq
import torch.ao.quantization as taq
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.nn.quantized as nnq
import torch.fx as fx

from evaluate import evaluate
from qconfigs import learnable_act, learnable_weights, fixed_act
from ipdb_hook import ipdb_sys_excepthook

from model.resnet import resnet18

ipdb_sys_excepthook()

# Intialize model
model = resnet18(pretrained=True)

# Define qconfigs
qconfig_FF = tq.QConfig(
    activation=learnable_act(range=2),
    weight=tq.default_fused_per_channel_wt_fake_quant
)

qconfig_QS_DQ = tq.QConfig(
    activation=fixed_act(min=0, max=1),
    weight=tq.default_fused_per_channel_wt_fake_quant
)


# Assign qconfigs
qconfig_mapping = QConfigMapping().set_object_type((taq.QuantStub, taq.DeQuantStub), qconfig_QS_DQ) \
                                .set_object_type(nnq.FloatFunctional, qconfig_FF)

# Awkward we have to do this manually, just for the sake of accessing the `out_channels` attribute
for name, module in model.named_modules():
    if hasattr(module, 'out_channels'):
        qconfig = tq.QConfig(
            activation=learnable_act(range=2),
            weight=learnable_weights(channels=module.out_channels)
        )
        qconfig_mapping.set_module_name(name, qconfig)
        #module.qconfig = qconfig


example_inputs = (torch.randn(1, 3, 224, 224),)

model.eval()
#import ipdb
#ipdb.set_trace()
# NOTE: check how the modules get fused, I want ConvReLU not ConvBNReLU
fx_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

# NOTE: need to figure out how to place the fixed qparams qconfig correctly
# at beginning and end. Also, mention that PTQ is on in both cases, so we are cheating
# by doing dynamic quant to some degree.
# `activation_post_process` is also moved outside the module, so each module has
# a `weight_fake_quant` attribute, but the `activation_post_process` is a seperate requantization
# step, and it isn't attached.
print('\n Original')
evaluate(model, 'cpu')

print('\n FX prepared')
evaluate(fx_model, 'cpu')

# Can experiment with visualize the graph, e.g.
# >> fx_model
# >> print(fx_model.graph)  # prints the DAG

# Prints the graph as a table
print("\nGraph as a Table:\n")
fx_model.graph.print_tabular()

# Plots the graph
# Need to install GraphViz and have it on PATH (or as a local PATH variable)
#from torch.fx import passes
#g = passes.graph_drawer.FxGraphDrawer(fx_model, 'fx-model')
#with open("graph.svg", "wb") as f:
    #f.write(g.get_dot_graph().create_svg())

# We will correct the graph to have fixedqparams on the input quantstub
# NOTE: taken from https://pytorch.org/docs/stable/fx.html#graph-manipulation
def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)

# Experiment with iterator pattern:
# NOTE: taken from https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
from torch.fx.node import Node
from typing import Dict

class GraphIteratorStorage:
    """
    A general Iterator over the graph. This class takes a `GraphModule`,
    and a callable `storage` representing a function that will store some
    attribute for each node when the `propagate` method is called.

    Its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments, e.g. an example input tensor.
    As each operation executes, the GraphIteratorStorage class stores
    away the result of the callable for the output values of each operation on
    the attributes of the operation's `Node`. For example,
    one could use a callable `store_shaped_dtype()` where:

    ```
    def store_shape_dtype(result):
        if isinstance(result, torch.Tensor):
            node.shape = result.shape
            node.dtype = result.dtype
    ```
    This would store the `shape` and `dtype` of each operation on
    its respective `Node`, for the given input to `propagate`.
    """
    def __init__(self, mod, storage):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.storage = storage

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to the `storage` function.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter
            self.storage(node, result)

            # Store the output activation in `env` for the given node
            env[node.name] = result

        #return load_arg(self.graph.result)

def store_shape_dtype(node, result):
    """
    Function that takes in the current node, and the tensor it is operating on (`result`)
    and stores the shape and dtype of `result` on the node as attributes.
    """
    if isinstance(result, torch.Tensor):
        node.shape = result.shape
        node.dtype = result.dtype
# NOTE: I just discovered that they have the `Interpreter` class, which accomplishes the same thing:
# https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter
GraphIteratorStorage(fx_model, store_shape_dtype).propagate(example_inputs[0])
for node in fx_model.graph.nodes:
    print(node.name, node.shape, node.dtype)



from torch.fx.node import Argument
from typing import Any, Dict, Tuple
from torch.fx import Transformer
class ConvBNReLUToConvReLU(Transformer):
    def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        return super().call_function(n)

    def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        if target == 'neg':
            call_self, *args_tail = args
            return call_self.sigmoid(*args_tail, **kwargs)
        return super().call_method(n)
    
    def call_module(self, target: 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        if 'conv' in target:
            orig = self.module.get_submodule(target)
            if type(orig) in [torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d]:
                call_module, *args_tail = args
                xxx
                # return new operation fed into call_module
        return super().call_module(target=target, args=args, kwargs=kwargs)


#gm = torch.fx.symbolic_trace(model)

transformed : torch.nn.Module = ConvBNReLUToConvReLU(fx_model).transform()
input = example_inputs[0]
out = transformed(input)
XXX