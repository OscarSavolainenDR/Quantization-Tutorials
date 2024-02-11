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
        This function defines a class BasicBlock for Neural Networks which takes
        several input parameters and sets member variables of the object accordingly:
        It also defines the methods to be called upon object creation completion
        i.e., constructor's syntax inside the function . These methods initialize
        member variables ,perform actions and set default behaviours for future uses.

        Args:
            inplanes (int): The `inplanes` input parameter of this `__init__`
                method indicates the number of input channels that will be processed
                by this block; specifically the first Convolution2d operation is
                passed this quantity and should have feature maps size as per
                output feature maps required .
            planes (int): The planes parameter is the number of output feature
                channels of each of the convolutional layers. This is not the
                number of input images; instead it’s an inherent property that
                defines what size block-filter must contain.
            stride (1): The `stride` parameter determines the step size of the
                convolution operation performed by the `conv3x3()` method within
                the `self.conv1` and `self.conv2` instances. A stride of 1 means
                that the convolution operates on the entire input image without downsampling.
            downsample (None): The downsample parameter specifies an optional
                feature map downsampling layer that is applied to the output of
                the first convolutional layer before the second one.
            groups (1): The `groups` input parameter of the `__init__()` method
                defines the number of split parts or groups that each convolutional
                layer is divided into. It controls how many parallel branches the
                layer has. When `groups != 1`, the basic block cannot support that
                setting and will throw a ValueError exception. By default `groups=1`,
                meaning the layer uses a single branch with a single convolutional
                layer and then a batch normalization layer directly after it.
            base_width (64): The `base_width` parameter defines the number of input
                channels to the first convolutional layer (32), and only supports
                a value of 64 for BasicBlock.
            dilation (1): The `dilation` parameter specifies whether to use dilated
                convolutions or standard convolutions. currently only supports
                standard convolutions (dilation=1).
            norm_layer (None): The `norm_layer` input parameter specifies the norm
                (or activation) function used for batch normalization inside the
                block.

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
        self.FFAddReLU = torch.ao.nn.quantized.FloatFunctional()
        # self.relu_out = nn.ReLU(inplace=True)
        self.stride = stride

    def modules_to_fuse(self, prefix):
        """
        This function takes a prefix argument and returns a list of tuples containing
        module names to be fused together. It targets the fusion of Convolutional
        layers with Batch Normalization and ReLU activations. The downsampling
        operation can also be included if required.

        Args:
            prefix (str): The `prefix` input parameter is a string that is used
                to construct the module names within the list of modules to be fused.

        Returns:
            list: The function "modules_to_fuse" returns a list of lists titled
            "modules_to_fuse". This list contains arrays of module names ( strings)
            separated by '.'.  Each nested list consists of 3 or 4 items depending
            on the value of "downsample". In the output they are separated by a comma.

        """
        modules_to_fuse_ = []
        modules_to_fuse_.append([f'{prefix}.conv1', f'{prefix}.bn1', f'{prefix}.relu1'])
        modules_to_fuse_.append([f'{prefix}.conv2', f'{prefix}.bn2'])
        if self.downsample:
            modules_to_fuse_.append([f'{prefix}.downsample.0', f'{prefix}.downsample.1'])

        return modules_to_fuse_

    def forward(self, x: Tensor) -> Tensor:
        """
        This function performs the following operations on input tensor 'x':
        1/ Convolution (conv1 and conv2)
        2/ Batch normalization (bn1 and bn2)
        3/ ReLU activation (relu1)
        4/ Optionally downsamples the input using 'downsample' if it is not None.
        5/ Applies FFAddReLU operation on out and identity (if provided).
        6/ Return the final output.

        Args:
            x (Tensor): The `x` input parameter is passed into each convolutional
                operation.

        Returns:
            Tensor: The output returned by this function is `out`.

        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.FFAddReLU.add_relu(out, identity)
        # out = self.relu_out(out)

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
        This is a constructive method that establishes a ResNet block class for
        use with PyTorch geeks. It takes various inputs like block kind Block and
        layers lists of integers indicating the quantity of residual blocks to use
        at each stage (1–3). The layer list also specifies if a particular layer
        should have stride 2 (therefore dilated) instead of traditional stride.
        Various settings such as standard width per group norms use BN/ReLU vs.
        Swish and many others can be used to configure the module during its
        lifetime; there's no precondition or presumption made about this input
        object when passing it here - only those explicitly specified as arguments
        are acceptable/legal; no defaulting allowed (such defaulting leads directly
        towards pitfalls that bring unexpected surprises later)!

        Args:
            block (Type[BasicBlock]): The `block` input parameter defines the
                instance of the class `Bottleneck` or `BasicBlock` that this ResNet
                instance will be built from.
            layers (List[int]): The `layers` input parameter specifies a list of
                integers that defines the number of channels for each layer. The
                length of the list is 3 and corresponds to the three residual
                blocks within the ResNet. The integer value at index 'i' specifies
                that the i-th residual block will have the given number of channels
            num_classes (1000): The num_classes input is an integer value passed
                into the ResNet architecture definition. It is defining the number
                of classes for the output layers following the bottleneck blocks
                within the residual building block itself has multiple residual
                layers followed by average pooling.   This parameter influences
                only output of a specific resolution size after the specified
                quantity as defined here passes through its final layer.  The
                default value for the integer type is set to be equal to one
                thousand. However the number can be anything assigned prior as a
                function input to best support whatever projected application would
                follow for such architectural settings designed within its context
                using that version. Therefore this function allows the programmer
                more customizability beyond out of box options found elsewhere.
            zero_init_residual (False): Zero-init residual is an input parameter
                to the ResNet class and when set to "true" initially zeroes out
                the weights of a BN layer inside a specific kind of block within
                ResNet so that every residual block works identically and speeds
                up ResNet by 0.2% - 0.3%.
            groups (1): The groups input parameter specifies the number of groups
                to split the input into for better parallelization and better
                hardware usage on some devices.
            width_per_group (64): The `width_per_group` parameter determines how
                many output channels each group of channels (i.e., every eighth
                channel) should have; increasing this parameter is said to
                significantly improve performance with high-resolution inputs at
                no penalty at reduced-resolution inputs but can increase computation
                cost. Typically 64 works well for the default input resolution and
                larger values of `width_per_group` give diminishing returns for
                reduced computational cost (in that order).
            replace_stride_with_dilation (None): In this ResNet-like architecture
                initialization function `replace_stride_with_dilation`, is a
                3-element list (tuple) of boolean values indicating if the 2x2
                standard strides should be replaced with dilated convolution instead
                for each group of input channels respectively. This means that
                some groups might retain their default standard stride whereas
                other groups have their stride dilated instead resulting  i increased
                spatial resolution after some layers
            norm_layer (None): The `norm_layer` input parameter is used to specify
                a normalization layer (i.e., Batch Normalization or Group
                Normalization) that should be applied to each residual block within
                this network model.  It's set to None by default but can also take
                an explicitly passed instance of `nn.BatchNorm2d` if a non-default
                norm layer is preferred.

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
        This function creates a sequence of residual blocks from a single basic
        block and returns them as a sequential layer. It adjusts the stride and
        dilates the previous layers to accommodate more feature channels and
        increases depth of the network without changing the resolution of each block.

        Args:
            block (Type[BasicBlock]): The `block` parameter is an instance of a
                class that inherits from `nn.Module`, representing a residual block
                to be created by the `_make_layer` function.
            planes (int): In the provided function `_make_layer`, `planes` is an
                input parameter representing the number of output channels for
                each basic block. It determines the number of planes that the layer
                should have after it is applied.
            blocks (int): The `blocks` input parameter represents how many times
                the given `block` (i.e., a basic block) should be repeated. It
                controls the number of times the function's body is executed when
                looped over by `for _.range(1 , blocks)`
            stride (1): The `stride` input parameter to the `_make_layer` function
                controls the number of feature channels that are skipped within
                each spatial block when processing a layer. A larger value of
                stride effectively reduces the spatial resolution of the input
                while increasing the number of features captured. It defaults to
                1 for no strides at all.
            dilate (False): The dilate parameter increases the stride of the
                convolutions inside each "block" inside the Sequential stack. It
                effectively dilates or skips over some pixels within each block's
                convaolution when applying it.

        Returns:
            nn.Sequential: The function _make_layer returns a nn.Sequential
            containing multiple layers formed from blocks and downsampling layers.

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
        This function takes an object "self" as input and returns a list of tuples
        representing the layers that can be fused together using fusion operations.
        Each tuple contains the names of the modules that can be fused.

        Returns:
            list: The function "modually_to_fuse" returns a list of lists named
            "module_to_fuse_" which contains all the modules to be fusioned. Each
            inner list within the outer list represents a layer and contains a
            list of modules to be fusinoned for that layer.

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
        This function performs a sequence of operations on an input tensor `x`
        using a series of layers (modules), including convolutional layers with
        batch normalization and ReLU activation functions , as well as fully
        connected layers (fc). It finally returns the processed tensor `x`

        Args:
            x (Tensor): Here's the answer:
                
                The `x` parameter is the input Tensor for the module. It undergoes
                various transformations and is finally returned as the output of
                the module after being processed through a series of convolutional
                and fully-connected layers.

        Returns:
            Tensor: The output returned by this function is a tensor.

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
        This function implements a sequence of quantization and de-quantization
        operations on an input tensor 'x', using the methods '_forward_impl' and
        'dequant'. It first quantizes the input 'x', applies some additional
        implementation-specific forward operation '._forward_impl', then de-quantizes
        the result back to the original precision. The final output is returned.

        Args:
            x (Tensor): The `x` parameter is the input tensor that passes through
                the function and is subject to quantumization and dequantization
                before being returned. It is not modified after the return.

        Returns:
            Tensor: Based on the code provided:
            The output is x.

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
    This function creates a ResNet model given a block and number of layers as
    input parameters. It has the option to load pre-trained weights from a WeightsEnum
    object. If pre-trained weights are used then it overwrites the number of classes
    parameter of the ResNet model.

    Args:
        block (Type[BasicBlock]): The input `block` determines which kind of ResNet
            block is to be created within this ResNet network instance; therefore
            this function needs an `AbstractType[BasicBlock]`.
        layers (List[int]): The `layers` input parameter specifies the number of
            residual blocks to include at each scale level of the ResNet.
        weights (Optional[WeightsEnum]): Based on the code provided it appears
            that weights are a pretrained weight set that is to be loaded into the
            resnet if the weights are not none and also overides certain hyper
            parameters specified by the kwargs dictionary.
        progress (bool): The `progress` input parameter indicates whether to show
            a progress bar for the loading of the pretrained weights or not.
        	-*kwargs (Any): The **kwargs parameter is a dictionary of key-value pairs
            that can be passed to the ResNet constructor. It allows for any
            additional parameters or configuration options for the ResNet model
            that are not included as explicit inputs to the _resnet function.

    Returns:
        ResNet: Based on the code provided:
        The output of the `_resnet` function is a `ResNet` object

    """
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

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
