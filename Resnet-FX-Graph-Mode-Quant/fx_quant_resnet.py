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

model = resnet18(pretrained=True)

# from torch.ao.quantization import get_default_qconfig_mapping
# qconfig_mapping = get_default_qconfig_mapping("fbgemm")

#qconfig_layer = tq.QConfig(
    #activation=learnable_act(range=2),
    #weight=learnable_weights
#)

qconfig_FF = tq.QConfig(
    activation=learnable_act(range=2),
    weight=tq.default_fused_per_channel_wt_fake_quant
)

qconfig_QS_DQ = tq.QConfig(
    activation=fixed_act(min=0, max=1),
    weight=tq.default_fused_per_channel_wt_fake_quant
)


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
        module.qconfig = qconfig


example_inputs = (torch.randn(1, 3, 224, 224),)

model.eval()
fx_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

model.train()
fake_quant_model = tq.prepare_qat(model, inplace=False)

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

print('\n Eager mode prepared')
evaluate(fake_quant_model, 'cpu')
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
    """
    This function transforms a PyTorch module 'm' by tracing it using an instance
    of the tracer class (optional) and modifies some nodes of the traced graph to
    replace instances of 'torch.add' with 'torch.mul'. Finally return a GraphModule
    that wraps the original module and the transformed graph.

    Args:
        m (torch.nn.Module): The `m` input parameter is the module that we want
            to transform and instrument. It serves as the entry point for our
            tracing system and we apply transformations to it before returning a
            new modified version of it.
        tracer_class (fx.Tracer): The `tracer_class` parameter specifies a class
            that will trace the module's execution path during the transformation
            process. The tracer class is used to generate the fx.Graph representation
            of the original module.

    Returns:
        torch.nn.Module: The output returned by the function transform is a new
        Torch module that is based on the traced graph of the input module using
        FX to create it. This means the new torch.nn.Module has the modified node
        from original modules.

    """
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
    the attributes of the operation's `Node`. For exmaple,
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
        This function initializes an object of an unspecified class (likely a class
        representing a module or a graph) and assigns the values of the module's
        `graph` attribute and a dictionary of named modules to instance attributes.
        It also stores the storage passed as an argument.

        Args:
            mod (): The mod parameter passes the Module object to this class's
                constructor. This module object has the module dictionary and graph
                linked to it within the overall module graph.
            storage (dict): The `storage` input parameter is a dictionary that
                stores information about the current module being traversed. It
                is used to keep track of the modules that have already been processed
                during the traverse.

        """
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.storage = storage

    def propagate(self, *args):
        """
        This function implements a graph-based modular code execution environment
        using Python's torch.fx. The given code is likely a modified version of
        the default `run` function for torch.fx modules and includes storage
        functionality not present there by default.
        The modifications include the following:
        1/ Modules stored as dictionaries `self.modules`. Each dictionary contains
        keys like functions/target names and values which point towards node objects
        defined previously elsewhere within another scope using torch expressions
        2/ Methods to handle nodes of types 'get_attr', call_function' (also
        including recursive invocation)...as well call_method' which allows attribute
        access from arbitrary object
        3/ storage( ) function call to custom storage handling beyond default
        results of modules
        All things considered. This seems primarily tailored for storing variables
        within expressions used across modules instead being forced through entire
        computational graphs when calling a final fx () function/expression or
        similar. Instead results can stay inside environments where they were
        produced by smaller operations until explicitly retrieved from this
        environment before proceeding.

        """
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            """
            This function takes an argument `a` and returns a function that maps
            each name within `a` to a value from the `env` dictionary. Essentially
            allowing you to treat your arguments as a graph and use environment
            variables as part of that graph.

            Args:
                a (): The input parameter 'a' is an argument to be mapped through
                    a lambda function that retrieves values from the environment.

            Returns:
                : The function `load_arg` returns a Torch Graph Map operation that
                takes an argument 'a' and returns a tensor where each element is
                looked up from an environment 'env' using the name of the node to
                which the argument belongs. In other words ,it maps every element
                of input tensor with corresponding value from env dictionary . The
                output is tensor type .

            """
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            """
            This function fetches an attribute of a target object by iterating
            through the dots (`) separated string representation of the target
            name and returning the final attribute value. It raises a RuntimeError
            if any intervening attribute does not exist.

            Args:
                target (str): The `target` parameter is a string representing the
                    name of an attribute on some object (which can be a node), and
                    this function fetches that attribute value using a series of
                    splits and GetAttr calls if there is no RuntimeError caused
                    by nonexistent target attributes.

            Returns:
                str: The output returned by this function is `attr_itr`, which is
                the final value of the attribute chain.

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

xxx

