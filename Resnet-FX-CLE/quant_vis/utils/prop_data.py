import torch


def forward_and_backprop_an_image(model: torch.nn.Module):
    """
    Forward and backwards propagate an image of a dog.
    """
    import urllib

    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms

    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model
    input_batch.to("cpu")
    model.to("cpu")

    # Feed data through the model
    output = model(input_batch)

    # We backpropagate the gradients. We take the mean of the output, ensuring that
    # we backprop a scalar where all outputs are equally represented.
    output.mean().backward()
