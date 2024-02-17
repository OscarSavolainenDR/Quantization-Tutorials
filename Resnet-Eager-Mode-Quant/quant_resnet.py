import torch
from model.resnet import resnet18
from evaluate import evaluate

model = resnet18(pretrained=True)
#print(model)

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
backend = 'fbgemm'# if x86 else 'qnnpack'
qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

fused_model.qconfig = qconfig

# Step 4: Prepare for fake-quant
fused_model.train()
fake_quant_model = torch.quantization.prepare_qat(fused_model, inplace=False)

# Step 5: convert (true int8 model)
converted_model = torch.quantization.convert(fake_quant_model)


print('\nfloat model')
evaluate(model)

print('\nfused model')
evaluate(fused_model)

print('\nfake quant model')
evaluate(fake_quant_model)

print('\nconverted model')
evaluate(converted_model)
import ipdb
ipdb.set_trace()