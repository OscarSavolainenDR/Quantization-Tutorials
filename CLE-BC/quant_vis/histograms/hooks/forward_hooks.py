import torch
import torch.quantization._numeric_suite as ns
from ...utils.act_histogram import ActHistogram
from ...utils.hooks import is_model_quantizable
from utils.dotdict import dotdict

from ...settings import HIST_XMIN, HIST_XMAX, HIST_QUANT_BIN_RATIO

from typing import Callable, Union
from utils.logger import setup_logger

# Configure the logger
logger = setup_logger(__name__)


def activation_forward_histogram_hook(
    act_histogram: ActHistogram, name: str, qscheme: torch.qscheme, bit_res: int = 8
):
    """
    A pre-forward hook that measures the floating-point activation being fed into a quantization module.
    This hook calculates a histogram, with the bins given by the quantization module's qparams,
    and stores the histogram in a global class.
    If the histogram for the given quantization module has not yet been initialised, this hook initialises
    it as an entry in a dict. If it has been initialised, this hook adds to it.

    Therefore, as more and more data is fed throuhg the quantization module and this hook,
    the histogram will accumulate the frequencies of all of the binned values.

    activation_histogram_hook inputs:
    - act_histogram (ActHistogram): a dataclass instance that stores the activation histograms and hook handles.
    - name (str): the name of the module, and how its histogram will be stored in the dict.
    - qscheme (torch.qscheme): the qscheme of the quantization module.
    - bit_res (int): the quantization bit width of the tensor, e.g. 8 for int8.

    hook inputs:
    - module: the quantization module.
    - input: the activation fed to the quantization module.
    """

    def hook(module, input):
        # Ensure we are in eval mode, and ensure that this is not during a Shadow conversion check.
        if not module.training and type(module) is not ns.Shadow:

            # Get number of quantization bins from the quantization bit width
            qrange = 2**bit_res

            local_input = input[0].detach().cpu()

            # If the entry in the `act_histogram` dict has not been initialised, i.e. this is the first forward pass
            # for this module
            if name not in act_histogram.data:
                # We calculate the limits of the histogram. These are dependent on the qparams, as well as how
                # much "buffer" we want on either side of the quantization range, defined by `HIST_XMIN` and
                # `HIST_XMAX` and the qparams.
                hist_min_bin = (-HIST_XMIN * qrange - module.zero_point) * module.scale
                hist_max_bin = (
                    (HIST_XMAX + 1) * qrange - module.zero_point
                ) * module.scale

                # If symmetric quantization, we offset the range by half.
                if qscheme in (
                    torch.per_channel_symmetric,
                    torch.per_tensor_symmetric,
                ):
                    hist_min_bin -= qrange / 2 * module.scale
                    hist_max_bin -= qrange / 2 * module.scale

                # Create the histogram bins, with `HIST_QUANT_BIN_RATIO` histogram bins per quantization bin.
                hist_bins = (
                    torch.arange(
                        hist_min_bin.item(),
                        hist_max_bin.item(),
                        (module.scale / HIST_QUANT_BIN_RATIO).item(),
                    )
                    - (0.5 * module.scale / HIST_QUANT_BIN_RATIO).item()
                    # NOTE: offset by half a quant bin fraction, so that quantization centroids
                    # fall into the middle of a histogram bin.
                )
                # TODO: figure out a way to do this histogram on CUDA
                tensor_histogram = torch.histogram(local_input, bins=hist_bins)

                # Create a map between the histogram and values by using torch.bucketize()
                # The idea is to be able to map the gradients to the same histogram bins
                bin_indices = torch.bucketize(local_input, tensor_histogram.bin_edges)

                # Initialise stored histogram for this quant module
                stored_histogram = dotdict()
                stored_histogram.hist = tensor_histogram.hist
                stored_histogram.bin_edges = tensor_histogram.bin_edges
                stored_histogram.bin_indices = bin_indices

                # Store final dict in `act_histogram`
                act_histogram.data[name] = stored_histogram

            # This histogram entry for this quant module has already been intialised.
            else:
                # We use the stored histogram bins to bin the incoming activation, and add its
                # frequencies to the histogram.
                histogram = torch.histogram(
                    local_input,
                    bins=act_histogram.data[name].bin_edges.cpu(),
                )
                act_histogram.data[name].hist += histogram.hist

                # We overwrite the bin indices with the most recent bin indices
                bin_indices = torch.bucketize(local_input, histogram.bin_edges)
                act_histogram.data[name].bin_indices = bin_indices

    return hook


def add_activation_forward_hooks(
    model: torch.nn.Module,
    conditions_met: Union[Callable, None] = None,
    bit_res: int = 8,
):
    """
    This function adds forward activation hooks to the quantization modules in the model, if their names
    match any of the patterns in `act_histogram.accepted_module_name_patterns`.
    These hooks measure and store an aggregated histogram, with the bins defined by the quantization
    grid. This tells us how the activation data is distributed on the quantization grid.

    Inputs:
    - model (torch.nn.Module):      the model we will be adding hooks to.
    - conditions_met (Callable):    a function that returns True if the conditons are met for
                                    adding a hook to a module, and false otherwise. Defaults to None.
    - bit_res (int): the quantization bit width of the tensor, e.g. 8 for int8.

    Returns:
    - act_histograms (ActHistogram): A dataclass instance that contains the stored histograms
                                    and hook handles.
    """

    # If the conditons are met for adding hooks
    if not is_model_quantizable(model, "activation"):
        logger.warning(f"None of the model activations are quantizable")
        return

    logger.warning(
        f"\nAdding forward activation histogram hooks. This will significantly slow down the forward calls for "
        "the targetted modules."
    )

    # We intialise a new ActHistogram instance, which will be responsible for containing the
    # activation histogram data
    act_histograms = ActHistogram(data={}, hook_handles={})

    # Add activation-hist pre-forward hooks to the desired quantizable module
    for name, module in model.named_modules():
        if hasattr(module, "fake_quant_enabled") and "weight_fake_quant" not in name:
            if conditions_met and not conditions_met(module, name):
                logger.debug(
                    f"The conditons for adding an activation hook to module {name} were not met."
                )
                continue

            hook_handle = module.register_forward_pre_hook(
                activation_forward_histogram_hook(
                    act_histograms,
                    name,
                    module.qscheme,
                    bit_res=bit_res,
                )
            )
            # We store the hook handles so that we can remove the hooks once we have finished
            # accumulating the histograms.
            act_histograms.hook_handles[name] = hook_handle

    return act_histograms
