import torch
from model.resnet import resnet18

model = resnet18(pretrained=True)
print(model)

# Step 1: architecture changes
# QuantStubs (we will do FloatFunctionals later)
# Done

# Step 2: fuse modules (recommended but not necessary)
modules_to_list = model.modules_to_fuse()

# It will keep Batchnorm
model.eval()
# fused_model = torch.ao.quantization.fuse_modules_qat(model, modules_to_list)

# This will fuse BatchNorm weights into the preceding Conv
fused_model = torch.ao.quantization.fuse_modules(model, modules_to_list)

# Step 3: Assign qconfigs


# Step 4: Prepare for fake-quant


# Step 5: convert (true int8 model)


import ipdb
ipdb.set_trace()