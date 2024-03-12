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
        This function defines a BasicBlock module for use with the PyTorch tensor
        processing library. It initializes the BasicBlock class with variables
        including 'stride' and groups', as well as setting norm_layers etc.  The
        block implements convolution layers as well as pooling layers if stride
        is non one to allow down sampling the input tensor.. It can add ReLU
        activation functions using its "self.add_relu" property..

        Args:
            inplanes (int): The `inplanes` parameter is the number of input channels
                to the block; it is used as the first dimension of the input tensors
                for the conv2d layers within the module.
            planes (int): In the given Python code definition of the `__init__()`
                function for class BasicBlock below lines have comments
                   ```
                   self.conv1 = conv3x3(inplanes * group_num + group_id * rise *
                plane_dim //(this*groupsize*stride),  planes)
                    ```
                Here plane dimensions (`plane_dim`) is calculated based on planes
                and stride parameters as shown below :
                   ```
                   self.conv1 = conv3x3(inplanes * group_num + group_id * rise *
                plane_dim //(this*groupsize*stride),  planes)
                     """planedim //=( this * group_size * strides )""" <- This
                part here is depending on  `plannes` parameter . The comnet above
                So planes does effect how many feature maps would be present (not
                channels actually ) but that feature maps/planes per layer will
                result if it passes to next step
            stride (1): In this PyTorch function definition for the `BasicBlock`
                class constructor it controls the spatial downsampling factor for
                the first conv layer: Consecutive applications of this block with
                a non-identity stride will make the output feature map shape smaller
                than the input's shape and thus will implement downsampling/pooling.
            downsample (None): The `downsample` parameter defines a downsampling
                module to use whenever the input has a stride other than 1. It is
                currently unimplemented when dilation > 1 so if it's passed the
                function will raise an error
            groups (1): The groups parameter specifies how many parallel branches
                will be created within the block and split input channels accordingly.
                Each branch has a subset of channels from input channels that are
                concatenated after each group has undergone depthwise separation
                using 1x1 convolutions followed by Batch Normalization and ReLU
                activation. In other words., It splits the channels of the input
                into 'groups' and uses multiple parallel branches with depthwise
                separable convolutions for more efficient computation and reduced
                memory usage. This design encourages less computations at a faster
                rate with quantized functions that create fewer Flops or Memory
                Access compared to dense connectivity like traditional convolution
                layers found later downstream where much bigger feature plans
                result after successive basic block expansions beyond an extent
                leading dense computing on heavy datasets while compromising
                space/memory or Flops consumption (when computing per element).
            base_width (64): The base_width parameter specifies the number of
                output channels (64 by default). The channel number will determine
                how wide the layers are; smaller numbers increase depth but lower
                the memory requirement and computation per unit of depth; bigger
                ones result conversely.
            dilation (1): In this context the dilation parameter does not do
                anything because it is set to 1 and "raise NotImplementedError("Dilation
                > 1 not supported...")" if its value exceeds one so effectively
                nothing at all will happen to any incoming input regarding dilation
                and as such there's no point passing any value for dilation
                parameters that are not equal to 1 because this specific block
                does not support those values anyway.
            norm_layer (None): The norm_layer parameter of this constructor is
                passed an optional layer to use for normalizing (normalization)
                activation functions after the first convolution operation of this
                basic block or multi-resolution block within the residual network
                that may improve training time and final test score if appropriately
                implemented. If no value exists or if none are passed when
                instantiating BasicBlock class objects during design-time of the
                model and you omit the = None keyword parameter when calling this
                initializer with passing a normalization layer implementation such
                as the batch normalization variant used throughout many other
                residual network implementations: BatchNorm2d then will use batch
                normalization (affine-parameterized); however batch normalization
                will NOT work on all variants for normalizing the activations for
                a particular channel within feature space during every iteration
                of training/activating this class ( BasicBlock class ), so check
                if your passed layer input to constructor is of BatchNorm2d then
                consider not including the  = None part at initialization. For
                this constructor parameter the user could instead provide an
                implementation for Institute of Electrical and Electronics Engineers
                (InstEEE) layer normalization. You must specify something or you
                will use defaults that can degrade your accuracy for activation
                functions depending on channel count ( feature plane ) across this
                entire block structure or the entirety of your Multi-Residual
                Network( MRNET).

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
        This function takes a 'prefix' and creates a list of tuples containing
        names of modules to fuse. These are mostly convolutional layers followed
        by batch normalization and ReLU activation. Depending on downsampling
        feature flagged modules from downsample layer may also be included.

        Args:
            prefix (str): The `prefix` parameter specifies the prefix to be added
                to each name of the modules being fused. This is used to create
                meaningful names for the modules that are combined and will be
                useful during debugging.

        Returns:
            list: The output of the given function is a list containing multiple
            lists of strings representing the modules to be fused together.

        """
        modules_to_fuse_ = []
        modules_to_fuse_.append([f'{prefix}.conv1', f'{prefix}.bn1', f'{prefix}.relu1'])
        modules_to_fuse_.append([f'{prefix}.conv2', f'{prefix}.bn2'])
        if self.downsample:
            modules_to_fuse_.append([f'{prefix}.downsample.0', f'{prefix}.downsample.1'])

        return modules_to_fuse_

    def forward(self, x: Tensor) -> Tensor:
        """
        This function performs a convolutional layer using two convolutional layers
        with batch normalization and ReLU activation functions between them. It
        also includes an optional downsampling step.

        Args:
            x (Tensor): In this code snippet `x` is just an input tensor that is
                passed through a series of operations such as conv1 & batch
                normalization (bn1). No changes are made to x.

        Returns:
            Tensor: The output returned by the function "forward" is "out".

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
        This function defines a ResNet block that can be used to build a ResNet
        model. It takes various parameters such as the number of input channels;
        the number of classes; and options for zero-initializing the residual
        blocks. The function also includes various default settings such as norm
        layer and dilations and stores these settings as class attributes.

        Args:
            block (Type[BasicBlock]): The block input parameter takes a BasicBlock
                instance as its argument. It is used to create the different layers
                of the ResNet architecture (conv1 through layer4) using the Block
                module factory method. Each Block has a number of parameters that
                are passed into their corresponding instances when calling them
                with various arguments as defined above. These parameters can
                include kernel size and stride among other things necessary for
                constructing a ResNet model from scratch given its input block structure
            layers (List[int]): The `layers` input parameter specifies the individual
                residual blocks within the bottleneck layers to be replaced or
                altered during the definition of a MobileNetV2 network architecture;
                this parameter is required when setting the replace stride with
                dilation argument to True to indicate whether replacement of
                convolutional strides with dilated ones needs take place inside
                each block based on parameters provided here so they needn't appear
                repeatedly within its successor residual blocks but instead solely
                define whether these alternatives happen somewhere during formation
                processes for better adaptability at those sites according to
                requirement needs specified through `layers`.
            num_classes (1000): The `num_classes` parameter represents the number
                of output classes for the model. It determines the dimensionality
                of the output from the final linear layer (self.fc). Here the
                module is constructed to work with a specific number of classes
                and scales other parameters such as the width of the convolutional
                layers accordingly
            zero_init_residual (False): In the given code snippet 'zero_init_residual'
                is an optional input parameter which takes a boolean value and it
                affects the initialization of BN layers present residual blocks
                inside the function. It set all of them to zero which enables each
                residual block to act like an identity transform (i-e every branch
                starts with zeroes)  and improves model accuracy by 0.2%-0.3 %
                according arXiv:1706.02677 paper
                
                In simple words the 'zero_init_residual' has ability to turn of
                or on a toggle switch to set every bias layer present residual
                blocks initially to zero(if 'True' input)or no-op (no side effects
                if input is  'False')
            groups (1): The `groups` parameter specifies the number of groups to
                divide the input channels into for each residual block. Dividing
                the input channels reduces the number of parameters and computations
                required during training and testing at an overall scale factor
                cost—for example ,if groups =16 the input 52 is split evenly amongst
                  16 groups of  3 dimensions. Increasing groups leads to reduced
                parameters but slows down testing speed due to higher memory
                consumption caused by 32-wide layers required to perform calculations
            width_per_group (64): The `width_per_group` parameter is a per-group
                width that is passed to the residual block. Residual blocks typically
                use separate weight matrices for each group within the residual
                block itself. For a specified number of groups `groups`, this param
                specifies the individual widths associated with each group; the
                specified values will then be added elementwise (i.e., channelwise).
                Aside from any remaining `dilation` within the residual block which
                also spreads activations further apart across the depth and width
                within each spatial group so they don't overfit the small spatial
                region assigned to them; width_per_group determines that original
                depth/width assigned by its layer 1 until after pooling
            replace_stride_with_dilation (None): The `replace_stride_with_dilation`
                parameter is a 3-element tuple that specifies if we should replace
                2x2 strides with dilated convolutions instead for each element.
                If the value is None or an empty list`, no replacements will be
                made. When replacing happens.
            norm_layer (None): The `norm_layer` parameter specifies the normalization
                layer to use for batch normalization (default `nn.BatchNorm2d`).
                If not provided as an argument here. Then it is assumed `None`.

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
        This function creates a layer factory that builds a stack of layers from
        the given BasicBlock instance. Each block has some shared parameters (e.g.,
        kernel size and number of groups), allowing for a efficient construction
        of these layers. It is possible to add downsampling and norm layer
        normalization as well as controlling padding with stride or dilating factors
        when adding new planes to allow flexibility over input height.

        Args:
            block (Type[BasicBlock]): The "block" input parameter of the "_make_layer"
                function takes the type of a BasicBlock object as an argument. It
                allows the function to create layers using different block
                architectures and the passed BasicBlock instance will be used to
                construct that layer
            planes (int): In this particular Function `def _make_layer(self ...`the
                planes input represents the number of feature plans to create or
                produce after going through the block function; think of it like
                this; whenever we do some blocks you know that whenever these
                feature channels as one layer; here we produce (feature plane *
                blocks), therefore the more plans we produce the more layers and
                the more feature channels. In simpler words the input `planes`
                defines number or productioof the featire channel produced .
            blocks (int): The `blocks` parameter represents the number of layers
                within the residual block and determines how many times the block
                is repeated. Each iteration increases the number of planes.
            stride (1): The stride parameter determines how far the layers'
                convolutional operation moves along the input volume when applying
                the layer. For instance，if the layer applies a stride of 2 to a
                3x3x3 input tensor and does not perform downsampling，it would
                output every other point across the three axes(yielding a 3x3x16
                output). If you instead set stride equal to 2 with dilate equal
                to True as well，every second 'hole' from the original volume will
                be kept but spaced further apart after being upsampled later
                (e.g.,26 times faster spatial resolution reduction or more for
                that matter) . Finally if only setting stride=1 or using no dilating
                (setting it initially false or omitting this altogether),you don't
                alter the density of feature extraction , maintaining instead
                constant every step 'overlap' from original volume  to feature map
                space and upsampling later.
            dilate (False): In this function `dilate` controls the amount of
                dilatation that each layer should apply. It's a Boolean parameter
                with default value set to `False`, when set to `True` increases
                the strides of the layer by stride amount. When set to `False`,
                it keeps the default behavior of not applying any dilations

        Returns:
            nn.Sequential: The output of this function is a Sequential model.

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
        This function constructs a list of tuples containing the names of PyTorch
        modules to fuse (e.g., "conv1", "bn1", etc.), based on the structure of
        the neural network defined using the `self` context (i.e., an instance of
        a class with attributes `layer1`, `layer2`, `layer3`, and `layer4`). Specifically:
        - It first initializes an empty list `modules_to_fuse_`.
        - Then it appends a tuple containing the names of modules to fuse for the
        first layer (`conv1`, `bn1`, `relu`).
        Next is a loop over the remaining layers (`layer1`, `layer2`, and `layer3`)
        using an explicit list comprehension and the eval() function to extract
        each layer as an instance of some class (probably self.layerX):
        - for each layer (in ['layer1', 'layer2', 'layer3', 'layer4'])
           obtains its module's modules to fuse (self.layerX[block_nb].modules_to_fuse(
        prefix ) ) by running eval() on the str() representation of the layer.
        - extends the initial list with each iteration result.
        Finally it returns a list of tuples containing the names of PyTorch modules
        to fuse after the loop completes its iteration over self.layer1 up to and
        including self.layer4 .
        In sum: it defines which PyTorch modules to fuse for the layers (up until
        layer 4) within the given neural network object instance so that they can
        be efficiently executed on various devices supported by FusePool2

        Returns:
            list: The function "modules_to_fuse" takes a self-object as input and
            returns a list of lists with the names of the layers (as strings) that
            are to be fused together. The list of layers is generated by iterating
            through the layers of the model and fetching their module fusion
            information using eval() to fetch attributes of the layers using string
            names

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
        This function performs the following operations on an input tensor 'x':
        	- Convolution using conv1()
        	- Batch Normalization using bn1()
        	- ReLU activation using relu()
        	- Max pooling using maxpool()
        	- Three more convolutional layers (layer1(), layer2(), and layer3())
        followed by batch normalization (bn4()) and ReLU activation (relu4())
        	- Average pooling (avgpool())
        	- Flattening (torch.flatten())
        	- Fully connected layer (fc())
        It returns the output of the fully connected layer.

        Args:
            x (Tensor): In this TorchScript implementation of a Neural Network
                class `_forward_impl()` method; `x` is the input Tensor to be
                processed. It first passes through successive layers like `conv1()`,
                `bn1()`, `relu()`, `maxpool()` and 3 further layer modules
                (`layer1()`, `layer2()`, and `layer3()`). Later on it runs avg
                pool () on it , followed by flatten(()) , then the fully connected(fc())
                layer and return the output.
                In simple terms , the `x` input parameter serves as the input to
                the entire Neural Network and gets passed through all its layers
                sequentially for forward propagation

        Returns:
            Tensor: The output of the given function is a tensor.

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
        This function implements a neural network layer. It first quantizes the
        input tensor "x" using a custom implementation called _forward_impl(), and
        then applies some computation. Finally it de-quantizes the output and
        returns it.

        Args:
            x (Tensor): Here's what the "x" parameter does based on the given code
                snippet:
                
                	- The function takes one argument 'x', which is a Tensor object.
                	- It goes through a sequence of modifications before being returned
                at the end of the forward function
                	- These transformations involve calling methods named _forward_impl()
                and dequant()
                
                In brief , "x" acts as the input that is processed and transformed
                by the forward function before being returned .

        Returns:
            Tensor: Based on the provided function signature `def forward(self`',
            the output returned by this function would be `Tensor`.

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
    This function creates a ResNet architecture using the given 'block' and 'layers',
    and returns the created ResNet model. If weights are provided as an optional
    parameter 'weights', it overwrites the 'num_classes' parameter of the model
    with the number of categories present' weighs.'meta["categories"]'.
    Finally the ResNet model is loaded from the provided state dict if weights are
    given or remains empty if not.

    Args:
        block (Type[BasicBlock]): The `_resnet` function's `block` parameter is
            an input argument that defines the block architecture for constructing
            ResNet layers using `BasicBlock`. The BasicBlock object serves as the
            building component for the deeper network architecture by chaining
            multiple blocks together with their output features concatenated and
            passing them through another version of the block for a faster
            optimization process.
        layers (List[int]): The `layers` input parameter specifies a list of integer
            values that determines the number of times each ResNet block is repeated.
        weights (Optional[WeightsEnum]): Based on the code snippet provided and
            without knowledge of any context beyond what is within the text itself:
            "If weights is not None" indicates that an optional input parameter
            named `weights`, which seems to be an object that should otherwise
            remain undefined here is being used conditionally - therefore;
            -weights appear(s) to refer( to) a collection of categories/subclass
            identifications by the attribute `metas`.
            In other words this line if(weights exist(s))..means it might load
            state_dict depending on weights.."category" and not from progress but
            as implied it means "classes".
        progress (bool): The `progress` parameter will display a progress bar
            during the loading of state dictionaries.
        	-*kwargs (Any): The **kwargs input parameter is a catch-all parameter
            that allows other parameters to be passed to the ResNet model's
            constructor. Any additional arguments passed to the function will be
            extracted and passed to the ResNet model as keyword arguments.

    Returns:
        ResNet: The output of the function `_resnet` is an instance of the `ResNet`
        class.

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
