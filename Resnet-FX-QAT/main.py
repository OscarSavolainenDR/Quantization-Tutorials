from typing import Tuple, Any, Union, List
from copy import deepcopy

import torch
import torch.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping

from evaluate import evaluate
from utils.qconfigs import learnable_act, learnable_weights, fake_quant_act
from utils.ipdb_hook import ipdb_sys_excepthook
from utils.graph_manip import convbn_to_conv

from model.resnet import resnet18

# Adds ipdb breakpoint if and where we have an error
ipdb_sys_excepthook()

# Intialize model
model = resnet18(pretrained=True)

# Define qconfigs
qconfig_global = tq.QConfig(
    activation=fake_quant_act,
    weight=tq.default_fused_per_channel_wt_fake_quant
)


# Assign qconfigs
qconfig_mapping = QConfigMapping().set_global(qconfig_global)

# We loop through the modules so that we can access the `out_channels` attribute
for name, module in model.named_modules():
    if hasattr(module, 'out_channels'):
        qconfig = tq.QConfig(
            activation=learnable_act(range=2),
            weight=learnable_weights(channels=module.out_channels)
        )
        qconfig_mapping.set_module_name(name, qconfig)

# Do symbolic tracing and quantization
example_inputs = (torch.randn(1, 3, 224, 224),)
model.eval()
fx_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

# Evaluate model
print('\nOriginal')
evaluate(model, 'cpu', 'Samoyed')

print('\nFX prepared')
evaluate(fx_model, 'cpu', 'Samoyed')

# Prints the graph as a table
print("\nGraph as a Table:\n")
fx_model.graph.print_tabular()

# Fuses Batchnorms into preceding convs
transformed : torch.fx.GraphModule = convbn_to_conv(deepcopy(fx_model))
input = example_inputs[0]
out = transformed(input) # Test we can feed something through the model
print('\nTransformed model evaluation:')
evaluate(transformed, 'cpu', 'Samoyed')



###############################
# Quantization Aware Training #
###############################
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from PIL import Image
from torchvision import transforms
from pathlib import Path

optim = Adam(transformed.parameters(), lr=1e-5)
loss_fn = CrossEntropyLoss()

# Used to get the index of the target image in the imagenet classes
def find_row_with_string(file_path, target_string):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file):
            if target_string in line:
                return line_number
    return None  # String not found in any row

def batch_images(targets: List[str], images_path: str, labels_path: Path):
    """
    Takes image labels (e.g. 'Samoyed'), and batches the processed image tensor together.
    It also produces a batched one-hot tensor, with the different images across the batch dimension.
    I.e., given a list of image names, it produces a batch of processed images and their onehot labels.
    """
    first_image = True
    for target in targets:
        # Get label index to create onehot vector
        row_number = find_row_with_string(labels_path, target)
        one_hot_label = torch.zeros(1000)
        one_hot_label[row_number] = 1

        # Get image
        input_image = Image.open(Path(f"{images_path}/{target}.jpg"))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)

        # Batch image and labels
        if first_image:
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            label_batch = one_hot_label.unsqueeze(0)
            first_image = False
        else:
            input_batch = torch.cat((input_batch, input_tensor.unsqueeze(0)), dim=0)
            label_batch = torch.cat((label_batch, one_hot_label.unsqueeze(0)), dim=0)

    return input_batch, label_batch

# Get input and ouput batches
images_path = Path("evaluate/images") # Path to the images
labels_path = Path('evaluate/imagenet_classes.txt')  # Path to your text file with imagenet class labels
batched_images, batched_labels = batch_images(['hen', 'Samoyed'], images_path=images_path, labels_path=labels_path)


def print_scale_and_zp(model, module_name):
    module = model.get_submodule(module_name)
    scale = module.scale
    zero_point = module.zero_point
    print(f"{module_name} scale and zero_point: {scale.item():.5}, {zero_point.item()}")

# Print out some paramaters before we do QAT
print('\nBefore QAT:')
print_scale_and_zp(transformed, 'activation_post_process_0')

# Training loop where we do QAT
mean_loss, counter = 0, 0
log_freq = 10
print('\nTraining loop')
for epoch in range(50):
    y_pred = transformed(batched_images)
    probabilities = torch.nn.functional.softmax(y_pred, dim=1)
    
    loss = loss_fn(probabilities, batched_labels)
    optim.zero_grad()
    loss.backward()
    optim.step()

    counter += 1
    if counter % log_freq  == 0:
        mean_loss += loss.item()
        print(f"Iter: {counter}, Mean loss: {(mean_loss/log_freq):.5}")
        mean_loss = 0

print('\nAfter QAT:')
print_scale_and_zp(transformed, 'activation_post_process_0')

# Post QAT evaluations
print('QAT model evaluation (Samoyed):')
evaluate(transformed, 'cpu', 'Samoyed')

# Check performance on hen
print('QAT model evaluation (hen):')
evaluate(transformed, 'cpu', 'hen')

# Check performance on clog (which we did not overfit to)
print('QAT model evaluation (clog):')
evaluate(transformed, 'cpu', 'clog')

# Check performance on clog (which we did not overfit to)
print('QAT model evaluation (mail box):')
evaluate(transformed, 'cpu', 'mail_box')
XXX
