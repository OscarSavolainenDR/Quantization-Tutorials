import os
from pathlib import Path

from quant_vis.histograms import (
    add_sensitivity_analysis_hooks,
    plot_quant_act_SA_hist,
    plot_quant_weight_hist,
)
import torch
from torch import fx
from torch.ao.quantization._equalize import equalize
from torch.ao.quantization._learnable_fake_quantize import (
    _LearnableFakeQuantize as LearnableFakeQuantize,
)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx
import torch.quantization as tq
from utils.ipdb_hook import ipdb_sys_excepthook

from evaluate import evaluate
from model.resnet import resnet18
from quant_vis.utils.prop_data import forward_and_backprop_an_image
from utils.graph_manip import (
    float_convbn_to_conv,
    qat_convbn_to_conv,
    get_previous_module_node,
)
from utils.qconfigs import (
    fake_quant_act,
    fake_quant_weight,
    learnable_act,
    learnable_weights,
)

# Adds ipdb breakpoint if and where we have an error
ipdb_sys_excepthook()

# Intialize model
model = resnet18(pretrained=True)


############################
# CROSS LAYER EQUALIZATION #
############################

# Graph-trace the model
model.train()
float_traced_model = fx.symbolic_trace(model)

# Merge all batchnorms into preceding convs
float_traced_model.eval()
float_traced_model = float_convbn_to_conv(float_traced_model)

# Iterate through graph, find CLE layer pairs.
# The graph is alrady in order of execution, and there isn't
# any complicated branching, so we can just treat all layers as
# sequential.
pairs = []
for node in float_traced_model.graph.nodes:
    if node.op == "call_module":
        module = float_traced_model.get_submodule(node.target)
        if hasattr(module, "weight"):
            prev_node = get_previous_module_node(
                float_traced_model,
                node,
                (torch.nn.Conv2d, torch.nn.Linear),
                CLE_compatible=True,
            )
            if prev_node:
                pairs.append([prev_node.target, node.target])

# Perform CLE from torch.ao.quantization._equalize import equalize
cle_model = equalize(float_traced_model, pairs, threshold=1e-4, inplace=False)

######################
# QUANTIZE THE MODEL #
######################
# Define qconfigs
qconfig_global = tq.QConfig(activation=fake_quant_act, weight=fake_quant_weight)

# Assign qconfigs
qconfig_mapping = QConfigMapping()

# We loop through the modules so that we can access the `out_channels` attribute
for name, module in cle_model.named_modules():
    if hasattr(module, "out_channels"):
        qconfig = tq.QConfig(
            activation=learnable_act(range=2),
            weight=learnable_weights(channels=module.out_channels),
        )
        qconfig_mapping.set_module_name(name, qconfig)


# Do symbolic tracing and quantization
example_inputs = (torch.randn(1, 3, 224, 224),)
cle_model.eval()
fx_model_w_cle = prepare_qat_fx(cle_model, qconfig_mapping, example_inputs)

# For comparison, we also get an FX model without CLE. We do so by
# performing FX quantization and fusing the BNs into the Convs.
qconfig_mapping = QConfigMapping()  # .set_global(qconfig_global)
for name, module in model.named_modules():
    if hasattr(module, "out_channels"):
        qconfig = tq.QConfig(
            activation=learnable_act(range=2),
            weight=learnable_weights(channels=module.out_channels),
        )
        qconfig_mapping.set_module_name(name, qconfig)
fx_model_no_cle = prepare_qat_fx(model, qconfig_mapping, example_inputs)
fx_model_no_cle.eval()
fx_model_no_cle = qat_convbn_to_conv(fx_model_no_cle)

# Evaluate model
print("\nOriginal")
evaluate(model, "cpu", "Samoyed")

print("\nTraced model")
evaluate(float_traced_model, "cpu", "Samoyed")

print("\nCLE model")
evaluate(cle_model, "cpu", "Samoyed")

print("\nFX prepared, with CLE")
evaluate(fx_model_w_cle, "cpu", "Samoyed")

# Check performance on hen
print("CLE model evaluation (hen):")
evaluate(fx_model_w_cle, "cpu", "hen")

# Check performance on clog (which we did not overfit to)
print("CLE model evaluation (clog):")
evaluate(fx_model_w_cle, "cpu", "clog")

# Check performance on clog (which we did not overfit to)
print("CLE model evaluation (mail box):")
evaluate(fx_model_w_cle, "cpu", "mail_box")

print("\nFX prepared, without CLE")
evaluate(fx_model_no_cle, "cpu", "Samoyed")

# Check performance on hen
print("FX prepared, without CLE (hen):")
evaluate(fx_model_no_cle, "cpu", "hen")

# Check performance on clog (which we did not overfit to)
print("FX prepared, without CLE (clog):")
evaluate(fx_model_no_cle, "cpu", "clog")

# Check performance on clog (which we did not overfit to)
print("FX prepared, without CLE (mail box):")
evaluate(fx_model_no_cle, "cpu", "mail_box")
# # Prints the graph as a table
# print("\nGraph as a Table:\n")
# fx_model.graph.print_tabular()

########################
# VISUALIZE CLE EFFECT #
########################


# ACTIVATION PLOTS
def create_act_plots(model, title):
    def conditions_met_forward_act_hook(module: torch.nn.Module, name: str) -> bool:
        if isinstance(module, LearnableFakeQuantize):
            # if '1' in name:
            print(f"Adding hook to {name}")
            return True
        return False

    # We add the hooks
    act_forward_histograms, act_backward_histograms = add_sensitivity_analysis_hooks(
        model, conditions_met=conditions_met_forward_act_hook, bit_res=8
    )

    forward_and_backprop_an_image(model)

    # Generate the forward and Sensitivity Analysis plots
    plot_quant_act_SA_hist(
        act_forward_histograms,
        act_backward_histograms,
        file_path=Path(os.path.abspath("") + f"/Histogram_plots/{title}"),
        sum_pos_1=[0.18, 0.60, 0.1, 0.1],  # location of the first mean intra-bin plot
        sum_pos_2=[0.75, 0.43, 0.1, 0.1],
        plot_title="SA act hists",
        module_name_mapping=None,
        bit_res=8,  # This should match the quantization resolution. Changing this will not change the model quantization, only the plots.
    )


# WEIGHT PLOTS
# Clear gradients with the sake of an otherwised unused optimizer
def create_weight_plots(model, title):
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=1)
    optimizer.zero_grad()

    # Check gradients cleared
    for parameter in model.parameters():
        assert parameter.grad is None

    # Produce new gradients
    forward_and_backprop_an_image(model)

    # Check gradients exist
    for parameter in model.parameters():
        assert parameter.grad is not None

    # Create the weight histogram plots, this time with Sensitivity Analysis
    # plots
    plot_quant_weight_hist(
        model,
        file_path=Path(os.path.abspath("") + f"/Histogram_plots/{title}"),
        plot_title="SA weight hists",
        module_name_mapping=None,
        conditions_met=None,
        # The below flag specifies that we should also do a Sensitivity
        # Analysis
        sensitivity_analysis=True,
    )


# # Cross Layer Equalized plots
# create_act_plots(fx_model_w_cle, "FX, with CLE")
# create_weight_plots(fx_model_w_cle, "FX, with CLE")
#
# # Original, non-Cross Layer Equalized plots
# create_act_plots(fx_model_no_cle, "FX, no CLE")
# create_weight_plots(fx_model_no_cle, "FX, no CLE")


###############
# EVALUATIONS #
###############
# Print out some paramaters before we do CLE
def print_scale_and_zp(model: torch.nn.Module, module_name: str):
    module = model.get_submodule(module_name)
    scale = module.scale
    zero_point = module.zero_point
    if len(scale) == 1:
        print(
            f"{module_name} scale and zero_point: {scale.item():.5}, {zero_point.item()}"
        )
    else:
        print(f"{module_name} scale and zero_point: {scale}, {zero_point}")


print("\nWithout CLE:")
print_scale_and_zp(fx_model_no_cle, "layer2.0.conv1.weight_fake_quant")

print("\nAfter CLE:")
print_scale_and_zp(fx_model_w_cle, "layer2.0.conv1.weight_fake_quant")

XXX
