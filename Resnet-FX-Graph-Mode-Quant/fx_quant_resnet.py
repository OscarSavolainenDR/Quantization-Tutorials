from typing import Tuple, Any, Union
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

# Adds ipdb breakpoint if and where we have an error
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


#########################
# SOME GRAPH TECHNIQUES #
#########################
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
        """
        This function initalizes an object of a class by setting its instance
        variables (member variables) to values from the parameter passed to the
        constructor. These include "self.mod" (which appears to be an object
        representing a module), "self.graph" and "self.storage", and a dictionary
        of modules named within that module ("self.modules").

        Args:
            mod (): The `mod` input parameter is passed by the user to the `__init__`
                method of the `MetaModel` class; it refers to a Modulen Object.
                In this context "module" denotes an object encapsulating code
                associated with particular Python modules or packages; for example:
                numpy. Num is the name of one such module.  The mod argument to
                the `init` method stores references both to a given package's /
                modules' underlying `graph`, (a structure of objects), and
                `named_modules` attribute which yields its constituent modules as
                a dictionary for future fast access; thus enabling convenient
                referencing of code regions within it when creating/querying
                metadata about its subcomponents and connections during computation
            storage (dict): The storage parameter stores a dictionary of Python
                module objects representing Python modules that have been imported
                at the time the cache was built and saved for future access to
                prevent further reimporting.

        """
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.storage = storage

    def propagate(self, *args):
        """
        This function takes a PyTorch FX graph object as input and executes it
        with a generic module interpreter. It recursively walks through the graph
        nodes and performs the following operations:
        1/ Maps arguments to their corresponding node's input_tensors.
        2/ Retrieves an attribute from a target node.
        3/ Calls a function/method with argument wrapping.
        4/ Stores output activations of each node to an env dictionary for later
        retrieval.

        """
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            """
            This function fetches an attribute from a given target string by
            splitting it into a list of atoms (segments), checking if each segment
            is a valid attribute of the current object using "hasattr()", and
            returning the value of the final attributed object.

            Args:
                target (str): The `target` parameter is a string that specifies
                    which attribute to fetch. The function splits the string into
                    a list of atoms (i.e., elements) using the dot character `.`
                    as the separator. It then iterates over these atoms and uses
                    each one as a keyword to access attributes on an object `attr_itr`.

            Returns:
                : The output of this function would be None if it encounters a
                runtime error because one or more referenced target atoms don't
                exist.  The fetch_attr function will throw a RuntimeError whenever
                the reference targets an undefined object property (the atom portion
                of the splited string does not exist), with error messages provided
                where applicable (including parts of the string prior to failing
                components)

            """
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



###########################################
### Fusing Bn in ConvBnReLU into ConvReLU #
###########################################
import copy
from torch.ao.nn.intrinsic.qat.modules.conv_fused import ConvBnReLU2d, ConvReLU2d, ConvBn2d
from torch.ao.nn.qat import Conv2d

def fuse_conv_bn_relu_eval(conv: Union[ConvBnReLU2d, ConvBn2d]) -> Union[ConvReLU2d, Conv2d]:
    """
    Given a quantizable ConvBnReLU2d Module returns a quantizable ConvReLU2d
    module such that the BatchNorm has been fused into the Conv, in inference mode.
    Given a ConvBn2d, it does the same to produce a Conv2d.
    """
    assert(not (conv.training or conv.bn.training)), "Fusion only for eval!"
    qconfig = conv.qconfig
    if type(conv) is ConvBnReLU2d:
        new_conv = ConvReLU2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                            conv.stride, conv.padding, conv.dilation,
                            conv.groups, conv.bias is not None,
                            conv.padding_mode, qconfig=qconfig)
    elif type(conv) is ConvBn2d:
        new_conv = Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                            conv.stride, conv.padding, conv.dilation,
                            conv.groups, conv.bias is not None,
                            conv.padding_mode, qconfig=qconfig)


    new_conv.weight, new_conv.bias = \
        fuse_conv_bn_weights(conv.weight, conv.bias,
                             conv.bn.running_mean, conv.bn.running_var, conv.bn.eps, conv.bn.weight, conv.bn.bias)

    return new_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """
    Helper function for fusing a Conv and BatchNorm into a single weight/bias tensor pair.
    """
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

# Graph manipulation functions for fusing Convs and BatchNorms
def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    """
    Helper function for having ` new_mdoule` take the place of `node`  in a dict of modules.
    """
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)
    
def convbnrelu_to_convrelu(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Iterates through the graph nodes, and:
    - where it finds a ConvBnReLU it replaces it with ConvReLU
    - where it finds a ConvBn it replaces it with Conv

    Inputs:
    fx_model (torch.fx.GraphModule): a graph module, that we want to perform transformations on

    Output:
    (torch.fx.GraphModule): a model where we have swapped out the 2d ConvBn/ConvBnReLU for Conv/ConvReLU, and
                            fused the Bns into the Convs.
    """
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for node in new_graph.nodes:
        # If the operation the node is doing is to call a module
        if node.op == 'call_module':
            # The current node
            orig = fx_model.get_submodule(node.target)
            if type(orig) in [ConvBnReLU2d, ConvBn2d]:
                # Produce a fused Bn equivalent.
                fused_conv = fuse_conv_bn_relu_eval(orig)
                # This updates `modules` so that `fused_conv`` takes the place of what was represented by `node`
                replace_node_module(node, modules, fused_conv)
    
    return fx.GraphModule(fx_model, new_graph)


transformed : torch.fx.GraphModule = convbnrelu_to_convrelu(fx_model)
input = example_inputs[0]
out = transformed(input) # Test we cna feed something through the model
print('\nTransformed model')
evaluate(transformed, 'cpu')
XXX