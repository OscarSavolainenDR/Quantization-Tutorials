#########################
# SOME GRAPH TECHNIQUES #
#########################
# Experiment with iterator pattern:
# NOTE: taken from https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
import torch
from torch import fx
from torch.fx.node import Node
from typing import Dict, Union, Tuple, Any

from torch.ao.nn.intrinsic.qat.modules.conv_fused import (
    ConvBnReLU2d,
    ConvReLU2d,
    ConvBn2d,
)
from torch.nn.modules.conv import Conv2d
from torch.ao.nn.qat import Conv2d as QATConv2d
from torch.nn.modules.batchnorm import BatchNorm2d


#########################################
# Fusing Bn in ConvBnReLU into ConvReLU #
#########################################
def qat_fuse_conv_bn_relu_eval(
    conv: Union[ConvBnReLU2d, ConvBn2d]
) -> Union[ConvReLU2d, Conv2d]:
    """
    Given a quantizable ConvBnReLU2d Module returns a quantizable ConvReLU2d
    module such that the BatchNorm has been fused into the Conv, in inference mode.
    Given a ConvBn2d, it does the same to produce a Conv2d.
    One could also use `torch.nn.utils.fuse_conv_bn_eval` to produce a Conv, and then quantize that as desired.
    """
    assert not (conv.training or conv.bn.training), "Fusion only for eval!"
    qconfig = conv.qconfig
    if type(conv) is ConvBnReLU2d:
        new_conv = ConvReLU2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            qconfig=qconfig,
        )
    elif type(conv) is ConvBn2d:
        new_conv = QATConv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            qconfig=qconfig,
        )
    else:
        raise NotImplementedError(f"conv type {type(conv)} not supported.")

    new_conv.weight, new_conv.bias = fuse_conv_bn_weights(
        conv.weight,
        conv.bias,
        conv.bn.running_mean,
        conv.bn.running_var,
        conv.bn.eps,
        conv.bn.weight,
        conv.bn.bias,
    )

    return new_conv


def float_fuse_conv_bn_relu_eval(
    conv: Union[ConvReLU2d, Conv2d], bn
) -> Union[ConvReLU2d, Conv2d]:
    """
    Given a Conv2d and a BatchNorm module pair, returns a Conv2d
    module such that the BatchNorm has been fused into the Conv, in inference mode.
    """
    assert not (conv.training or bn.training), "Fusion only for eval!"
    if type(conv) is Conv2d:
        new_conv = Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
        )

    new_conv.weight, new_conv.bias = fuse_conv_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

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

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(conv_w.shape) - 1)
    )
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


# Graph manipulation functions for fusing Convs and BatchNorms
def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_node_module(
    node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
):
    """
    Helper function for having `new_module` take the place of `node` in a dict of modules.
    """
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    # modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def replace_conv_bn_pair(
    conv_node: fx.Node,
    bn_node: fx.Node,
    modules: Dict[str, Any],
    new_module: torch.nn.Module,
    model: torch.fx.GraphModule,
):
    """
    Helper function for having `new_module` take the place of two adjacent
    nodes (`conv_node` and `bn_node`) in a dict of modules.
    """
    # Replace the Convs with Convs with fused Batchnorms
    assert isinstance(conv_node.target, str)
    parent_name, name = _parent_name(conv_node.target)
    modules[conv_node.target] = new_module
    setattr(modules[parent_name], name, new_module)

    # Delete the Batchnorms from the graph
    assert isinstance(bn_node.target, str)
    bn_node.replace_all_uses_with(bn_node.args[0])
    model.graph.erase_node(bn_node)


def get_previous_module_node(
    fx_model: torch.fx.GraphModule,
    node: torch.fx.Node,
    module_type,
    CLE_compatible: bool = False,
):
    """
    For a given node, find the closest previous node of a certain type.
    module_type: Can be an individual module type, or a tuple of types.

    If we specify that `CLE_compatible = True`, we only return a predecessor if
    the nodes between the current node and its predecessor don't have any
    CLE-breaking operations between them, e.g. `avgpool`, `add`, etc.
    """
    modules = dict(fx_model.named_modules())

    # Traverse the graph backwards
    for predecessor in node.all_input_nodes:
        if isinstance(predecessor, torch.fx.Node):

            # Return None if we run into any CLE breaking operation
            if CLE_compatible:
                if predecessor.target not in modules:
                    return None

                # If the current node is CLE "breaking", we didn't find a
                # predecessor that was CLE compatible
                if not isinstance(
                    fx_model.get_submodule(node.target), torch.nn.ReLU
                ) and not isinstance(fx_model.get_submodule(node.target), module_type):
                    return None

            # If we found the preceding node that matches the type
            if predecessor.target in modules:
                # Check if the predecessor node is the desired module type
                if isinstance(fx_model.get_submodule(predecessor.target), module_type):
                    return predecessor

            # Recursively search for the module in the predecessor's inputs
            prev_module_node = get_previous_module_node(
                fx_model, predecessor, module_type, CLE_compatible
            )
            if prev_module_node is not None:
                return prev_module_node

    # Module not found in the previous nodes
    return None


def qat_convbn_to_conv(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Iterates through the graph nodes, and:
    - where it finds a ConvBnReLU it replaces it with ConvReLU
    - where it finds a ConvBn it replaces it with Conv

    This function works in-place on `fx_model`.

    Inputs:
    fx_model (torch.fx.GraphModule): a graph module, that we want to
                            perform transformations on.

    Output:
    (torch.fx.GraphModule): a model where we have swapped out the 2d
                            ConvBn/ConvBnReLU for Conv/ConvReLU, and
                            fused the Bns into the Convs.
    """
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        # If the operation the node is doing is to call a module
        if node.op == "call_module":
            # The current node
            orig = fx_model.get_submodule(node.target)
            if type(orig) in [ConvBnReLU2d, ConvBn2d]:
                # Produce a fused Bn equivalent.
                fused_conv = qat_fuse_conv_bn_relu_eval(orig)
                # This updates `modules` so that `fused_conv` takes the place
                # of what was represented by `node`
                replace_node_module(node, modules, fused_conv)

    return fx_model


def float_convbn_to_conv(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Iterates through the graph nodes, and where it finds a pair of
    Conv-Bn nodes it replaces it with Conv with the Bn fused in.

    This is distinct from the `qat_convbn_to_conv` function that deals with
    taking fused [ConvBnReLU2d, ConvBn2d] instances and replaces them with
    quantized Conv2d/ConvReLU2d equivalents.

    This function works in-place on `fx_model`.

    Inputs:
    fx_model (torch.fx.GraphModule): a graph module, that we want to
                            perform transformations on.

    Output:
    (torch.fx.GraphModule): a model where we have swapped out the 2d
                            ConvBn/ConvBnReLU for Conv/ConvReLU, and
                            fused the Bns into the Convs.
    """
    modules = dict(fx_model.named_modules())

    pair, pairs = [], []
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            module = fx_model.get_submodule(node.target)
            if hasattr(module, "weight"):
                pair.append(node)
                if len(pair) == 2:
                    if isinstance(module, BatchNorm2d):
                        pairs.append(pair)
                    pair = []
                    pair.append(node)

    for conv_bn in pairs:
        conv_name, bn_name = conv_bn
        conv = fx_model.get_submodule(conv_name.target)
        bn = fx_model.get_submodule(bn_name.target)
        # Produce a fused Bn equivalent.
        fused_conv = float_fuse_conv_bn_relu_eval(conv, bn)
        # This updates `modules` so that `fused_conv` takes the place of what
        # was represented by `node`
        replace_conv_bn_pair(conv_name, bn_name, modules, fused_conv, fx_model)

    return fx_model
