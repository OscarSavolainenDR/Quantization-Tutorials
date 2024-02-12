import torch
import torch.quantization as tq
import torch.ao.quantization as taq
from torch.ao.quantization.quantize_fx import prepare_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.nn.quantized as nnq

from evaluate import evaluate
from qconfigs import learnable_act, learnable_weights, fixed_act

from model.resnet import resnet18

model = resnet18(weights='resnet18_weights.default')

# from torch.ao.quantization import get_default_qconfig_mapping
# qconfig_mapping = get_default_qconfig_mapping("fbgemm")

qconfig_layer = tq.QConfig(
    activation=learnable_act(range=2),
    weight=learnable_weights
)

qconfig_FF = tq.QConfig(
    activation=learnable_act(range=2),
    weight=tq.default_fused_per_channel_wt_fake_quant
)

qconfig_QS_DQ = tq.QConfig(
    activation=fixed_act(min=0, max=1),
    weight=tq.default_fused_per_channel_wt_fake_quant
)

model = resnet18(pretrained=True)

qconfig_mapping = QConfigMapping().set_global(qconfig_layer) \
                                .set_object_type((taq.QuantStub, taq.DeQuantStub), qconfig_QS_DQ) \
                                .set_object_type(nnq.FloatFunctional, qconfig_FF)



example_inputs = (torch.randn(1, 3, 224, 224),)

model.eval()
prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)

prepared_model.train()
fake_quant_model = tq.prepare_qat(prepared_model, inplace=False)
prepared_model.eval()

print('\n Original')
evaluate(model, 'cpu')

print('\n FX prepared')
evaluate(prepared_model, 'cpu')

# Can experiment with visualize the graph, e.g.
# >> prepared_model
# >> print(prepared_model.graph)  # prints the DAG

# Plots the graph
# Need to install GraphViz and have it on PATH (or as a local PATH variable)
from torch.fx import passes
g = passes.graph_drawer.FxGraphDrawer(fake_quant_model, 'resnet18-fakequant')
with open("graph.svg", "wb") as f:
    f.write(g.get_dot_graph().create_svg())