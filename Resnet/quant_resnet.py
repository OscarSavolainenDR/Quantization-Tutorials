import torch
from model.resnet import resnet18

def evaluate(model, device_str: str):
    # Download an example image from the pytorch website
    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    # sample execution (requires torchvision)

    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available, or to CPU if converted
    if not (device_str in['cpu', 'cuda']):
        raise NotImplementedError("`device_str` should be 'cpu' or 'cuda' ")
    if device_str == 'cuda':
        assert torch.cuda.is_available(), 'Check CUDA is available'

    input_batch = input_batch.to(device_str)
    model.to(device_str)

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

model = resnet18(pretrained=True)
# print(model)

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
backend = 'qnnpack'
qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

for name, module in fused_model.named_modules():
    module.qconfig = qconfig

# Step 4: Prepare for fake-quant
fake_quant_model = torch.ao.quantization.prepare(fused_model)

# Evaluate
print('\noriginal')
evaluate(model, 'cuda')
print('\nfused')
evaluate(fused_model, 'cuda')


# Step 5: convert (true int8 model)
fake_quant_model.to('cpu')
converted_model = torch.quantization.convert(fake_quant_model)

print('\nfake quant')
evaluate(fake_quant_model, 'cuda')


print('\nconverted')
evaluate(converted_model, 'cpu')
