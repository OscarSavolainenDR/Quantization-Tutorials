import torch
import torch.quantization as tq
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization._learnable_fake_quantize import (
    _LearnableFakeQuantize as LearnableFakeQuantize,
)

learnable_act = lambda range : LearnableFakeQuantize.with_args(
    observer=tq.HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    scale=range / 255.0,
    zero_point=0.0,
    use_grad_scaling=True,
)

learnable_weights = lambda channels : LearnableFakeQuantize.with_args(  # need to specify number of channels here
    observer=tq.PerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    scale=0.1,
    zero_point=0.0,
    use_grad_scaling=True,
    channel_len=channels,
)

fake_quant_act =  FakeQuantize.with_args(
    observer=tq.HistogramObserver.with_args(
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
)