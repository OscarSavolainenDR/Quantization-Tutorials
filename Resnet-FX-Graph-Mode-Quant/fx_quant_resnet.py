import torch
import torch.quantization as tq
import torch.ao.quantization as taq
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.nn.quantized as nnq

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
xxx
# Plots the graph
# Need to install GraphViz and have it on PATH (or as a local PATH variable)
#from torch.fx import passes
#g = passes.graph_drawer.FxGraphDrawer(fx_model, 'fx-model')
#with open("graph.svg", "wb") as f:
    #f.write(g.get_dot_graph().create_svg())