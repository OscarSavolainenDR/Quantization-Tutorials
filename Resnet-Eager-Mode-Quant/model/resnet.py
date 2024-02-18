from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        This function defines a class BasicBlock that takes an integer `inplanes`,
        `planes`, `stride`, `downsample`, `groups`, and other hyperparameters as
        inputs to create a residual block with a shortcut connection to process data.
        It consists of several convolution layers( conv3x3) that help extract
        features from input data by increasing or decreasing the spatial resolution.
        These layers come after normalization functions such that the feature can
        retain all important details. The final result will contain less noise
        with helpful features; the residual connection preserves these.

        Args:
            inplanes (int): The `inplanes` input parameter specifies the number
                of channels of the input data.
            planes (int): The planes input parameter defines the number of feature
                channels within each of the two convolutional layers inside this
                functional module. It also sets how many output values come out
                of those convolutional layers and then is used as an input value
                for both batch normalization layers (which produce the activations
                from norm-ing/gating input through that batchnormalizer). It
                represents what kind or quantity comes into each batchnorm activation
                function--the activations themselves have no dependence upon this
                input parameter. Essentially--planes are how many "sensitive"
                values come into every batchnormalized function.
            stride (1): The `stride` input parameter to the `__init__()` method
                determines the stride of the convolution operation performed by
                the `self.conv1` layer. It defaults to 1 if not provided.
            downsample (None): The downsample input parameter specifies an optional
                module that performs downsampling on the input to the block. If
                none is provided (default value None), no downsampling is performed
                and thus stride=1 by default). Downsampling modules can also be
                further modified by defining one yourself using Optional[nn.Module],
                with this default option being no additional module at all.
            groups (1): The input `groups` determines how many inputs from previous
                layer should be bundled into each grouping for concatenation into
                subsequent layers. BasicBlocks only support grouping 1 and 64 base
                width channels which isn't explicitly handled. Any non-default
                group setting causes a value error.
            base_width (64): The base width represents the default kernel size.
                The parameter is usually set to a common value that helps reduce
                parameters. This parameter can also be adjusted when more detailed
                regularization mechanisms are introduced or the capacity is increased.
            dilation (1): The `dilation` input parameter controls the spatial
                dilation applied to the inputs during convolution. It determines
                how far the filtered output is shifted relative to the input when
                convolving. Specifically , if `dilation > 1`, it raises a
                NotImplementedError because dilation greater than one is not
                currently supported by this implementation of BasicBlock.
            norm_layer (None): The `norm_layer` parameter is an optional Callable
                that specifies the batch normalization layer to use when instantiating
                this `BasicBlock`. If `norm_layer` is not passed (i.e., `None`),
                then a default Batch Normalization with kernel size of 2D (3x3)
                is used by default.

        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.add_relu_FF = torch.ao.nn.quantized.FloatFunctional()

    def modules_to_fuse(self, prefix):
        """
        This function creates a list of tuples containing the names of modules
        that will be fused together during the fusion step of a neural network's
        forward pass.

        Args:
            prefix (str): The prefix input parameter is a string that defines the
                name prefix for all the layers that are being added to the fusion
                module. It is used to create the desired layer names.

        Returns:
            list: The function modules_to_fuse returns a list of lists called
            modules_to_fuse_, containing three sublists of names for possible fusion.

        """
        modules_to_fuse_ = []
        modules_to_fuse_.append([f'{prefix}.conv1', f'{prefix}.bn1', f'{prefix}.relu1'])
        modules_to_fuse_.append([f'{prefix}.conv2', f'{prefix}.bn2'])
        if self.downsample:
            modules_to_fuse_.append([f'{prefix}.downsample.0', f'{prefix}.downsample.1'])

        return modules_to_fuse_

    def forward(self, x: Tensor) -> Tensor:
        """
        This function performs a forward pass through a neural network. It first
        passes the input tensor through a series of convolutional and pooling
        layers (represented by self.conv1 and self.downsample), followed by some
        batch normalization and ReLU activation functions (self.bn1 and self.relu1).
        After that is completed then another series of similar layers are performed
        before finishing with the add_relu_FF.add_relu function which adds another
        convolutional layer and relu function to the output. In concise terms it's
        forwarding an input tensor through a complex network.

        Args:
            x (Tensor): Here's what x is:
                
                x is a Tensor input into the Neural Network module.

        Returns:
            Tensor: The output returned by the function is "out".

        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu_FF.add_relu(out, identity)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        This function defines a class called ResNet with components such as layers
        and blocks and takes input parameters such as block types and the number
        of classes.

        Args:
            block (Type[BasicBlock]): The `block` parameter is the class type that
                determines which kind of block to use when constructing the network
                architecture.
            layers (List[int]): The input parameter 'layers' is a list of integers
                representing the number of feature channels for each layer and
                controls the resolution and depth of the ResNet model.
            num_classes (1000): The num_classes parameter determines how many
                classes the fully connected (fc) layer at the end of the function
                should have. Specifically stated as such a large number that there
                should be no concern for running out of memory when calculating
                these values.
            zero_init_residual (False): The zero_init_residual parameter initially
                sets all of the weight tensors for residual blocks to zero so that
                each block functions like an identity map; otherwise normal residual
                connections with learned weight tensors will be computed. Zero
                initialization may improve models by about 0.2-0.3% as suggested
                by arxiv paper links embedded.
            groups (1): The `groups` parameter specifies the number of groups to
                which the layers will be split. The width of each group is set to
                `width_per_group`, and a 3D convolution will replace the standard
                convolution if it finds one with a stride larger than 2. This
                improves model performance by enabling parallel computations for
                better efficiency.
            width_per_group (64): The `width_per_group` input parameter of the
                ResNet constructore functions determines the number of output
                channels per group of blocks (instead of the entire layer). This
                allows for a more efficient use of parameters and computational resources.
            replace_stride_with_dilation (None): The `replace_stride_with_dilation`
                input parameter is a tuple of length 3 that specifies if the 2x2
                strides should be replaced with dilated convolutions instead. Each
                element of the tuple indicates if replacement should be done for
                one of the three spatial axes. The ` replace_stride_with_dilation[i]`
                value of True means to replace the strides with dilation for the
                ith spatial axis
            norm_layer (None): The norm_layer input parameter defines the norm
                (normalization) layer used throughout the model's residual blocks.

        """
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """
        This function creates a sequence of layers from a input plane number to
        output planes number with given blocks and stride using dilated convolutional
        blocks with norm layer.

        Args:
            block (Type[BasicBlock]): The `block` parameter is an object of type
                BasicBlock (whatever that is). It's used as a factory for creating
                layers using the block() method called on it. This appears to allow
                for different types of layers to be created conditionally based
                on what block type is passed to the function when defining the
                sequential module.
            planes (int): The "planes" parameter controls the number of feature
                channels for each spatial resolution plane (layer). A bigger number
                raises the amount of information preserved from the initial picture
                at every spatial resolution level; however because it directly
                affects computation complexity—multiplying planes with each increase
                does by the cube—keep it moderate or too big when compared to
                height and width.
            blocks (int): The blocks input parameter defines how many instances
                of the given block layer should be made. This is used to repeat
                the construction multiple times to increase depth.
            stride (1): The `stride` input parameter specifies the number of
                channels to move down a layer when applying the block. A stride
                of 2 would mean that every other channel is skipped over resulting
                and fewer feature maps at each level of downsampling
            dilate (False): The dilate parameter allows the layers to stride across
                more input space by doubling the amount of spatial stride taken
                with each downsampling. Essentially it skips one pixel every other
                pixel to be precise and moves twice as much input space per
                convolution operation instead of moving a single input space for
                each one.

        Returns:
            nn.Sequential: The output of the function is an nn.Sequential layer
            comprising multiple blocks.

        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )


        return nn.Sequential(*layers)

    def modules_to_fuse(self):
        """
        This function constructs a list of tuples that represents the modules to
        fuse inside each layer of the model and returns it. Each tuple consists
        of the module name and the name of the layer prefixed to it (e.g.,
        conv1.bn1.relu). The list is built by iterating through each layer's modules
        using eval to retrieve the relevant layers for that layer.

        Returns:
            list: The output returned by this function is a list called
            `modules_to_fuse_` containing sublists obtained by recursively running
            `module.modules_to_fuse(prefix)` for all modules within each layer of
            the model and concatenating the resulting lists.

        """
        modules_to_fuse_ = []
        modules_to_fuse_.append(['conv1', 'bn1', 'relu'])

        for layer_str in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = eval(f'self.{layer_str}')
            for block_nb in range(len(layer)):
                prefix = f'{layer_str}.{block_nb}'
                modules_to_fuse_layer = layer[block_nb].modules_to_fuse(prefix)
                modules_to_fuse_.extend(modules_to_fuse_layer)

        return modules_to_fuse_

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        """
        This function is a forward pass implementation for a neural network. It
        takes a tensor as input and applies multiple linear transformations and
        pooling operations to produce the output. The transformation includes
        convolutions (self.conv1 and self.layer1), batch normalization (self.bn1),
        ReLU activation (self.relu) and maximum pooling (self.maxpool). It then
        flattens the output and applies a fully connected layer (self.fc). Finally
        it returns the result as a tensor.

        Args:
            x (Tensor): In this function `x` is the input tensor that is being
                passed through the network. It is being modified and transformed
                step by step using different layers such as convolutional
                layers(conv1), batch normalization layers(bn1) and ReLU activation
                functions(relu). After every layer it is getting returned to the
                function `x = ...` so at the end `x` contains the final output of
                the network.

        Returns:
            Tensor: The output of this function is a tensor.

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        This function takes a tensor as input and applies two consecutive Quantization
        operations (with inverse operation later) to forward the tensor.

        Args:
            x (Tensor): In the context of this function definition , `x` is the
                input tensor being passed to the function. The function takes a
                single argument `x` which is a tensor of type `Tensor`.

        Returns:
            Tensor: The output returned by this function is the tensor x after it
            has been transformed via the forward function of the subclass
            implementation and then restored to its original type.

        """
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x


def _resnet(
    block: Type[BasicBlock],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    """
    This function creates a ResNet model using the specified BasicBlocks and number
    of layers. If weights are provided as an argument then the function overwrites
    the number of classes paramater of the model with the number of categories
    present n the weights metadata. Then the model is loaded from the weight
    sstate-dict using progressive loading if required.

    Args:
        block (Type[BasicBlock]): The `block` input parameter specifies the type
            of BasicBlock to use when creating layers within the ResNet. It controls
            the architecture of the network.
        layers (List[int]): The layers parameter is a list of integers representing
            the number of residual blocks to include and how many residual blocks
            are present before every bottleneck.
        weights (Optional[WeightsEnum]): Based on the code snippet provided; The
            weights input parameter allows the function to load pre-trained models.
            If specified and if the 'progress' argument is true ,the model will
            also print progress bars
        progress (bool): Based on the code provided within the ResNet function and
            assuming all functions that aren't defined locally to this snippet
            work correctly the variable progress simply tells this snippet if it
            should show "Loading" progress bar or not as per user preference.
        	-*kwargs (Any): In this specific instance of Python function definition
            syntax; ```**kwargs```) is used to represent a dictionary of other
            arbitrary named arguments which will be passed to another part of the
            code during execution that consumes and processes this Python class
            or method respectively . The purpose of **kwargs is two fold : Firstly
            its serves as catch-all for any additional parameter  whose name the
            function may not already know about but can still process  Secondly
            it is often seen used for convenience (or poor practice?), passing
            dict's that already hold arbitrary key value pairs rather than having
            to construct another manually iterative named arg dictionary; thereby
            skipping an extraneous construct or copy operation of existing data
            (which are likely shared by both source & final function anyway).
            In other words it essentially allows one to pass whatever 'random '
            name value pairs whose presence doesn't violate the strict calling spec.
            
            The above implementation uses it to accept arbitrary name -value
            key-values pairs  that can be utilized to set default or otherwise
            unspecified paramters  to ResNet instance

    Returns:
        ResNet: Based on the function's definition it returns an object of the
        class "ResNet".

    """
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_ops": 1.814,
            "_file_size": 44.661,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
