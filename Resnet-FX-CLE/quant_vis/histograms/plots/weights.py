import torch
from typing import Tuple, Union

from ...settings import HIST_QUANT_BIN_RATIO, HIST_XMAX, HIST_XMIN

from utils.logger import setup_logger

# Configure logger
logger = setup_logger(__name__)


def get_weight_quant_histogram(
    weight: torch.nn.Parameter,
    scale: torch.nn.Parameter,
    zero_point: torch.nn.Parameter,
    qscheme: torch.qscheme,
    sensitivity_analysis: bool = False,
    bit_res: int = 8,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Union[torch.tensor, None]]:
    """
    Calculates the histogram of the weight, with bins defined by its scale and zero-point.
    Unlike the activation, we plot the weight tensor on the integer scale. This is because:
    1) The weight tensor values are difficult to interpret anyway, so there isn't much to gain from the original scale.
    2) Normalizing by each channel's quantization parameters makes sense, so we can aggregate across channels.

    Inputs:
    - weight (torch.nn.Parameter): a weight tensor
    - scale (torch.nn.Parameter):  a qparam scale. This can be a single parameter, or in the case of per-channel quantization, a tensor with len > 1.
    - zero_point (torch.nn.Parameter): a qparam zero_point. This can be a single parameter, or in the case of per-channel quantization, a tensor with len > 1.
    - qscheme: specifies the quantization scheme of the weight tensor.
    - sensitivity_analysis (bool): whether ot nor, if we have grads, should we plot the sensitivity analysis for the weights
    - bit_res (int): the quantization bit width, e.g. 8 for 8-bit quantization.

    Outputs:
    - hist (Tuple[torch.Tensor, torch.Tensor]): torch.histogram output instance, with histogram and bin edges.
    """

    if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
        scale = scale.view(len(scale), 1, 1, 1)
        zero_point = zero_point.view(len(zero_point), 1, 1, 1)
    elif qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
        pass
    else:
        raise ValueError(
            "`qscheme` variable should be per-channel symmetric or affine, or per-tensor symmetric or affine"
        )

    # Weight tensor in fake-quantized space
    fake_quant_tensor = weight.detach() / scale.detach() + zero_point.detach()

    # Flatten the weight tensor
    fake_quant_tensor = fake_quant_tensor.reshape(-1)

    # Get number of quantization bins from the quantization bit width
    qrange = 2**bit_res

    # Calculate the histogram between `-HIST_XMIN * qrange` and `(1+HIST_MAX_XLIM) * qrange`, with `HIST_QUANT_BIN_RATIO` samples per quantization bin.
    # This covers space on either side of the 0-qrange quantization range, so we can see any overflow, i.e clamping.
    hist_bins = torch.arange(
        -HIST_XMIN * qrange,
        (1 + HIST_XMAX) * qrange,
        1 / HIST_QUANT_BIN_RATIO,
    )
    # If we are doing symmetric quantization, center the range at 0.
    if qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
        hist_bins -= qrange / 2

    fake_quant_tensor = fake_quant_tensor.cpu()
    hist = torch.histogram(fake_quant_tensor, bins=hist_bins)

    if sensitivity_analysis and weight.grad is not None:
        # Create a map between the histogram and values by using torch.bucketize()
        # The idea is to be able to map the gradients to the same histogram bins
        bin_indices = torch.bucketize(
            fake_quant_tensor, hist.bin_edges
        )  # , right=True)

        # Compute the sum of gradients, with the forward histogram bins, using torch.bincount()
        binned_grads = torch.bincount(
            bin_indices.flatten(), weights=weight.grad.flatten()
        )

        # Padding may be required, if the bin_indices (which stop at the index of the maximum value
        # of `fake_quant_tensor` when mapped on to the histogram) is smaller than the maximum
        # histogram bin value. I.e., if the tensor doesn't fill the rightmost bin of the histogram.
        size_diff = hist_bins.size()[0] - bin_indices.max() - 2
        if size_diff > 0:
            # We add zeros to the end, as bincount automatically zero-pads the beginning as needed
            padding = torch.zeros(size_diff)
            binned_grads = torch.concat([binned_grads, padding])
        elif size_diff == -2:
            binned_grads = binned_grads[1:-1]
        elif size_diff == -1:
            binned_grads = binned_grads[:-1]

        return hist, binned_grads

    elif sensitivity_analysis:
        logger.warning(
            "`get_weight_quant_histogram` provided `sensitivity_analysis=True`, but the weight tensor does not have any attached gradients."
        )

    return hist, None
