import os
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from pathlib import Path

from ...utils.act_histogram import ActHistogram
from ...settings import HIST_QUANT_BIN_RATIO, SMOOTH_WINDOW, SUM_POS_1_DEFAULT, SUM_POS_2_DEFAULT
from .weights import get_weight_quant_histogram
from .utils import (
    get_prob_mass_outside_quant_range,
    fill_in_mean_subplot,
    draw_centroids_and_tensor_range,
    moving_average,
    create_double_level_plot_folder,
)

from typing import Callable, Tuple, Dict, Union, List
from utils.logger import setup_logger

# Configure logger
logger = setup_logger(__name__)

def _plot_single_tensor_histogram(
    forward_hist: Tuple[torch.Tensor, torch.Tensor],
    filename: Path,
    params: Dict,
    sum_pos_1: List[float] = SUM_POS_1_DEFAULT, 
    bit_res: int = 8,
):
    """
    Helper function for plotting the histogram of a single quantized tensor, overlain with the quantization grid of the tensor.
    It is specifically designed for plotting weight/activation tensors, but can be adapted for other tensors.
    It plots the histogram locally as `.png` files.

    Inputs:
    - hist (Tuple[torch.Tensor, torch.Tensor]): a Torch histogram tuple (histogram and bin edges),
    - filename (Path): the filename that the histogram will be plotted to. This is expected to be an image file.
    - params (Dict): a dict containing metadata on the plot name.
    - sum_pos_1 (List[float]): coordinates for the sub-plot for the forward histogram
    - bit_res (int): the quantization bit width of the tensor, e.g. 8 for int8.

    Returns:
    None
    """

    #####################
    # DATA MANIPULATION #
    #####################

    bin_edges = forward_hist.bin_edges.cpu()
    qrange = 2**bit_res  # Number of quantization bins

    # Convert forward histogram to probability distribution
    forward_hist_pdf = (forward_hist.hist / torch.sum(forward_hist.hist)).cpu()
    # Extract zero-bin, and set forward histogram value to 0 to improve visibility of histogram
    zero_bin_index = torch.argmin(torch.abs(bin_edges))
    zero_bin_forward_value = copy.deepcopy(
        torch.sum(forward_hist_pdf[zero_bin_index - 1 : zero_bin_index + 1])
    )
    forward_hist_pdf[zero_bin_index - 1 : zero_bin_index + 1] = 0
    # Measure forward clamped prob mass
    clamped_forward_prob_mass = get_prob_mass_outside_quant_range(
        forward_hist_pdf, qrange
    )

    ####################
    # PREPARE THE PLOT #
    ####################
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Remove the grid
    ax.grid(visible=False)

    # Quantization scale, extracted from histogram bin width
    scale = (
        forward_hist.bin_edges[1] - forward_hist.bin_edges[0]
    ) * HIST_QUANT_BIN_RATIO
    scale = scale.cpu()

    # Find where the histogram value is greater than zero, as this gives
    # the dynamic range of the floating point tensor.
    nonzero_indices = torch.nonzero(forward_hist.hist > 0).squeeze()

    # Get the dynamic range of the floating point tensor, i.e. the smallest and largest indices where the histogram value > 0
    tensor_min_index = nonzero_indices.min().item()
    tensor_max_index = nonzero_indices.max().item()

    # Adds red lines representing quantization centroids, and black lines representing the floating-point
    # tensor dynamic range.
    draw_centroids_and_tensor_range(
        ax, bin_edges, qrange, tensor_min_index, tensor_max_index, scale
    )

    # Plot the CDF/Histogram of the forward histogram and gradients
    markersize = plt.rcParams["lines.markersize"] / 4  # Set the desired marker size
    if len(bin_edges) == len(forward_hist_pdf):
        trimmed_bin_edges = bin_edges
    else:
        trimmed_bin_edges = bin_edges[:-1]

    # Forward histogram
    ax.plot(
        trimmed_bin_edges,
        forward_hist_pdf,
        "b.",
        markersize=markersize,
        label=params["act_or_weight"],
    )

    # Add smoothed out plot for the histogram (`SMOOTH_WINDOW` quantization bins)
    smoothed_hist = moving_average(forward_hist_pdf, window_size=HIST_QUANT_BIN_RATIO*SMOOTH_WINDOW)

    # Smoothed forward histogram
    ax.plot(
        trimmed_bin_edges,
        smoothed_hist,
        "b-",
        markersize=markersize,
        label=f"Smoothed {params["act_or_weight"]}.",
    )

    # Plot labels
    plt.title(
        f"{params['act_or_weight']} - {params['title']} - {params['module_name']}"
        if params["title"]
        else f"{params['act_or_weight']} - {params['module_name']}"
    )
    # Depending on if we're plotting weights or activations, we customise the xlabel
    if params["act_or_weight"] == "Weight":
        plt.xlabel(
            f"{params['act_or_weight']} value on an integer scale, overlain with quantization bins"
        )
    elif params["act_or_weight"] == "Activation":
        plt.xlabel(f"{params['act_or_weight']} value, overlain with quantization bins")
    else:
        raise ValueError(
            "the `params` `act_or_weight` value should be `Activation` or `Weight`"
        )
    plt.ylabel(f"Probability of {params['act_or_weight']} value")
    plt.legend(loc="upper right")

    ########################################
    # Add average intra-bin behavior plots #
    ########################################
    # Create the overlay subplot, a small plot at the top left of the figure
    ax_sub = fig.add_axes(sum_pos_1)

    # Fill in the mini intra-bin behavior / summary statistics plot
    fill_in_mean_subplot(
        forward_hist_pdf,
        zero_bin_forward_value,
        clamped_forward_prob_mass,
        ax_sub,
        color="b",
        data_name=params["act_or_weight"],
    )
    
    # Save fig with high resolution
    # NOTE: dpi = dots per inch, where a smaller value means faster plot generation but less resolution
    fig.savefig(filename, dpi=450)
    plt.close()


def plot_quant_act_hist(
    act_forward_histograms: ActHistogram,
    file_path: Path,
    plot_title: Union[str, None] = None,
    module_name_mapping: Union[Callable, None] = None,
    sum_pos_1: List[float] = [0.18, 0.60, 0.1, 0.1], 
    bit_res: int = 8,
):
    """
    When called, this function plots the activation histograms. These histograms contain aggregated
    forward histogram data for whatever modules the hooks have been added to, specified in `act_forward_histograms`.

    Inputs:
    - act_forward_histograms (ActHistogram): dataclass instance with forward activation histograms and hook handles
    - file_path (Path): path to the folder in which the files should be plotted.
    - plot_title (str): title given to the plot, which will also include the module's name.
    - module_name_mapping (Union[Callable, None]): a function that edits the name of the module to whatever alias is desired. Default to None.
    - sum_pos_1 (List[float]): coordinates for the sub-plot for the forward histogram
    - bit_res (int): the quantization bit width, e.g. 8 for 8-bit quantization.
    """

    # If activation histograms are empty, ignore
    if not act_forward_histograms.data:
        logger.warning(f"\nNo activation quant histograms were stored.")
        return

    # Store module metadata in a dict
    params = {}
    params["title"] = plot_title
    log_msg = f"\nPlotting quantized activation tensor histograms"
    log_msg = log_msg if not plot_title else log_msg + f" for {plot_title}"
    log_msg += ". This can take a long time."
    logger.info(log_msg)

    # Create plotting folders
    act_plot_folder = create_double_level_plot_folder(file_path, 'activations', 'hists')

    # For each module
    for module_name in act_forward_histograms.data:
        histogram = act_forward_histograms.data[module_name]

        # We have the option of using some alias for the module name in the plot title.
        if module_name_mapping:
            params["module_name"] = module_name_mapping(module_name)
        else:
            params["module_name"] = module_name

        # Activation histograms
        params["act_or_weight"] = "Activation"
        activation_plot_filename = (
            act_plot_folder / f"Act-hist-{params['module_name']}.png"
        )
        _plot_single_tensor_histogram(
            histogram, activation_plot_filename, params, sum_pos_1, bit_res
        )

        # Remove hook once we're done plotting the histogram
        # Otherwise, the hook will remain in, slowing down forward calls
        handle = act_forward_histograms.hook_handles[module_name]
        handle.remove()

    # We reset the activation histogram object
    act_forward_histograms.reset()


##########################
# WEIGHT HISTOGRAM PLOTS #
##########################
def plot_quant_weight_hist(
    model: torch.nn.Module,
    file_path: Path,
    plot_title: Union[str, None] = None,
    module_name_mapping: Union[Callable, None] = None,
    conditions_met: Union[Callable, None] = None,
    sum_pos_1: List[float] = SUM_POS_1_DEFAULT, 
    sum_pos_2: List[float] = SUM_POS_2_DEFAULT,
    sensitivity_analysis: bool = False,
    bit_res: int = 8,
):
    """
    When called, this function plots the histogram of the  models weight tensors, overlain with the quantization grid.

    Each weight tensor is scaled so that:
            quant_weight_tensor = weight_tensor/scale + zero_point.
    This is also true in the case of per-channel quantization of weights.

    Inputs:
    - model (torch.nn.Module): the model whose weight histograms we are plotting.
    - file_path (Path): path to the folder in which the files should be plotted.
    - plot_title (str): title given to the plot, which will also include the module's name.
    - module_name_mapping (Callable): a function that edits the name of the module to whatever alias is desired.
    - conditions_met (Callable): a function that returns True if the conditons are met for
                                adding a hook to a module, and false otherwise. Defaults to None.
    - sum_pos_1 (List[float]): coordinates for the sub-plot for the weight tensor histogram
    - sum_pos_2 (List[float]): coordinates for the sub-plot for the gradients
    - sensitivity_analysis (bool): whether ot nor, if we have grads for the weight tensor, 
                                should we plot the sensitivity analysis for the weights.
    - bit_res (int): the quantization bit width of the tensor, e.g. 8 for int8.
    """

    assert isinstance(
        file_path, Path
    ), "`file_path` variable should be of type pathlib.Path"

    # Store module metadata in a dict
    params = {}
    params["title"] = plot_title
    log_msg = f"\nPlotting quantized weight tensor histograms"
    log_msg = log_msg if not plot_title else log_msg + f" for {plot_title}"
    log_msg += ". This can take a long time."
    logger.info(log_msg)

    # For each module
    for module_name, module in model.named_modules():

        # Some modules don't have weights, e.g. QuantStubs.
        if hasattr(module, "weight_fake_quant"):
            # Check if the conditions were met for this module
            if conditions_met and not conditions_met(module, module_name):
                logger.debug(
                    f"The conditons for plotting the histogram for the weight tensor of module {module_name} were not met."
                )
                continue

            # We get the weight histogram from the tensor and its qparams
            weight_histogram, binned_weight_grads = get_weight_quant_histogram(
                module.weight,
                module.weight_fake_quant.scale,
                module.weight_fake_quant.zero_point,
                module.weight_fake_quant.qscheme,
                sensitivity_analysis,
                bit_res,
            )

            # We have the option of using some alias for the module name in the plot title.
            if module_name_mapping:
                params["module_name"] = module_name_mapping(module_name)
            else:
                params["module_name"] = module_name

            # Weight histograms
            params["act_or_weight"] = "Weight"

            # Plot the weight histogram
            if sensitivity_analysis and binned_weight_grads is not None:
                # Create folder
                weight_plot_folder = create_double_level_plot_folder(file_path, "weights", "sensitivity_analysis")
                weight_plot_filename = weight_plot_folder / f"Weight-hist-{params['module_name']}.png"
                
                # Generate plots
                _plot_SA_tensor_histogram(
                    weight_histogram,
                    binned_weight_grads,
                    weight_plot_filename,
                    params=params,
                    sum_pos_1=sum_pos_1,
                    sum_pos_2=sum_pos_2,
                    bit_res=bit_res,
                )
            elif sensitivity_analysis:
                logger.warning(f"`plot_quant_weight_hist` provided `sensitivity_analysis=True`, but no weight tensor binned gradients were provided for module {params['module_name']}.")        
            else:
                # No gradients were found, or we are not doing a sensitivity analysis for the weights
                # Create folder
                weight_plot_folder = create_double_level_plot_folder(file_path, "weights", "hists")
                weight_plot_filename = weight_plot_folder / f"Weight-hist-{params['module_name']}.png"

                # Generate plots
                _plot_single_tensor_histogram(
                    weight_histogram, weight_plot_filename, params, sum_pos_1, bit_res
                )


##############################
# SENSITIVITY ANALYSIS PLOTS #
##############################
def plot_quant_act_SA_hist(
    act_forward_histograms: ActHistogram,
    act_backward_histograms: ActHistogram,
    file_path: Path,
    sum_pos_1: List[float] = SUM_POS_1_DEFAULT, 
    sum_pos_2: List[float] = SUM_POS_2_DEFAULT,
    plot_title: Union[str, None] = None,
    module_name_mapping: Union[Callable, None] = None,
    bit_res: int = 8,
):
    """
    When called, this function plots the sensitivity analysis activation histograms. These histograms contain two
    things: 1) aggregated forward histogram data for whatever modules the hooks have been added to, specified in
    `act_forward_histograms`, and 2) the sensitivity analysis overlain onto the quantization grid. The former (forward)
    tells us the distribution of the data, and how it interacts with the quantization grid. The latter (backward)
    tells us the "importance" of each quantization bin for the given input.

    Inputs:
    - act_forward_histograms (ActHistogram): dataclass instance with forward activation histograms and hook handles
    - file_path (Path): path to the folder in which the files should be plotted.
    - sum_pos_1 (List[float]): coordinates for the sub-plot for the forward histogram
    - sum_pos_2 (List[float]): coordinates for the sub-plot for the gradients
    - plot_title (str): title given to the plot, which will also include the module's name.
    - module_name_mapping (Union[Callable, None]): a function that edits the name of the module to whatever alias is desired. Default to None.
    - bit_res (int): the quantization bit width, e.g. 8 for 8-bit quantization.
    """

    # If activation histograms are empty, ignore
    if not act_forward_histograms.data:
        logger.warning(f"\nNo activation quant histograms were stored.")
        return

    # Store module metadata in a dict
    params = {}
    params["title"] = plot_title
    log_msg = f"\nPlotting quantization sensitivity analysis plots"
    log_msg = log_msg if not plot_title else log_msg + f" for {plot_title}"
    log_msg += ". This can take a long time."
    logger.info(log_msg)

    # Create plotting folders
    act_plot_folder = create_double_level_plot_folder(file_path, 'activations', 'sensitivity_analysis')

    # For each module
    for module_name in act_forward_histograms.data:
        for_histogram = act_forward_histograms.data[module_name]
        back_binned_grads = act_backward_histograms.data[module_name].binned_grads

        # We have the option of using some alias for the module name in the plot title.
        if module_name_mapping:
            params["module_name"] = module_name_mapping(module_name)
        else:
            params["module_name"] = module_name

        # Activation histograms
        params["act_or_weight"] = "Activation"
        activation_plot_filename = (
            act_plot_folder / f"SA-Act-hist-{params['module_name']}.png"
        )
        _plot_SA_tensor_histogram(
            for_histogram,
            back_binned_grads,
            activation_plot_filename,
            params,
            sum_pos_1,
            sum_pos_2,
            bit_res,
        )

    # We reset the activation histogram objects. This also removes the associated hooks.
    act_forward_histograms.reset()
    act_backward_histograms.reset()


def _plot_SA_tensor_histogram(
    forward_hist: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    binned_back_grads: torch.Tensor,
    filename: Path,
    params: Dict,
    sum_pos_1: List[float] = SUM_POS_1_DEFAULT, 
    sum_pos_2: List[float] = SUM_POS_2_DEFAULT,
    bit_res: int = 8,
):
    """
    Function for plotting the histogram of a quantized tensor, overlain with the quantization grid of the tensor.
    This includes the forward and SA plots.

    Inputs:
    - forward_hist (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): a Torch histogram tuple (histogram, bin edges and bin_indices),
    - binned_back_grads (torch.Tensor): summed gradients, corresponding to the histogram bin edges.
    - filename (Path): the filename that the histogram will be plotted to. This is expected to be an image file.
    - params (Dict): a dict containing metadata on the plot name.
    - sum_pos_1 (List[float]): coordinates for the sub-plot for the forward histogram
    - sum_pos_2 (List[float]): coordinates for the sub-plot for the gradients
    - bit_res (int): the quantization bit width of the tensor, e.g. 8 for int8.

    Returns:
    None
    """
    #####################
    # DATA MANIPULATION #
    #####################

    bin_edges = forward_hist.bin_edges.cpu()
    qrange = 2**bit_res  # Number of quantization bins

    # Convert forward histogram to probability distribution
    forward_hist_pdf = (forward_hist.hist / torch.sum(forward_hist.hist)).cpu()
    # Extract zero-bin, and set forward histogram value to 0 to improve visibility of histogram
    zero_bin_index = torch.argmin(torch.abs(bin_edges))
    zero_bin_forward_value = copy.deepcopy(
        torch.sum(forward_hist_pdf[zero_bin_index - 1 : zero_bin_index + 1])
    )
    forward_hist_pdf[zero_bin_index - 1 : zero_bin_index + 1] = 0
    # Measure forward clamped prob mass
    clamped_forward_prob_mass = get_prob_mass_outside_quant_range(
        forward_hist_pdf, qrange
    )

    # Do the same for the gradients:
    # Convert to PDF
    grad_hist_pdf = torch.abs(binned_back_grads)
    grad_hist_pdf = grad_hist_pdf / torch.sum(grad_hist_pdf)
    # Extra zero-bin value
    zero_bin_grad_value = copy.deepcopy(
        torch.sum(grad_hist_pdf[zero_bin_index - 1 : zero_bin_index + 1])
    )
    grad_hist_pdf[zero_bin_index - 1 : zero_bin_index + 1] = 0
    # Clamped prob mass
    clamped_grad_prob_mass = get_prob_mass_outside_quant_range(grad_hist_pdf, qrange)

    ####################
    # PREPARE THE PLOT #
    ####################
    # We use the histogram bin widths to access the quantization scale
    scale = (
        forward_hist.bin_edges[1] - forward_hist.bin_edges[0]
    ) * HIST_QUANT_BIN_RATIO
    scale = scale.cpu()

    # Find the dynamic range of the floating tensor, i.e. the min and max indices where the histogram value > 0
    nonzero_indices = torch.nonzero(forward_hist.hist > 0).squeeze()
    tensor_min_index = nonzero_indices.min().item()
    tensor_max_index = nonzero_indices.max().item()

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Remove the grid
    ax.grid(visible=False)

    # Adds red lines representing quantization centroids, and black lines representing the floating-point
    # tensor dynamic range.
    draw_centroids_and_tensor_range(
        ax, bin_edges, qrange, tensor_min_index, tensor_max_index, scale
    )

    # Plot the CDF/Histogram of the forward histogram and gradients
    markersize = plt.rcParams["lines.markersize"] / 4  # Set the desired marker size
    if len(bin_edges) == len(forward_hist_pdf):
        trimmed_bin_edges = bin_edges
    else:
        trimmed_bin_edges = bin_edges[:-1]

    # Forward histogram
    ax.plot(
        trimmed_bin_edges,
        forward_hist_pdf,
        "b.",
        markersize=markersize,
        label=f"Tensor Probability",
    )
    # Gradients
    ax.plot(
        trimmed_bin_edges,
        grad_hist_pdf,
        "g.",
        markersize=markersize,
        label="Grad Probability",
    )

    # Add smoothed out plot for the histogram and gradients (`SMOOTH_WINDOW` quantization bins)
    smoothed_hist = moving_average(forward_hist_pdf, window_size=HIST_QUANT_BIN_RATIO*SMOOTH_WINDOW)
    smoothed_grad = moving_average(grad_hist_pdf, window_size=HIST_QUANT_BIN_RATIO*SMOOTH_WINDOW)

    # Smoothed forward histogram
    ax.plot(
        trimmed_bin_edges,
        smoothed_hist,
        "b-",
        markersize=markersize,
        label=f"Smoothed Tensor Prob.",
    )
    # Smoothed Gradients
    ax.plot(
        trimmed_bin_edges,
        smoothed_grad,
        "g-",
        markersize=markersize,
        label="Smoothed Grad Prob.",
    )

    # Plot labels
    plt.title(f"Sensitivity Analysis - {params['title']} - {params['module_name']}")
    plt.xlabel(f"{params["act_or_weight"]}/Gradient value, overlain with quantization bins")
    plt.ylabel(f"Probability")
    plt.legend(loc="upper right")

    ########################################
    # Add average intra-bin behavior plots #
    ########################################

    ################
    # FORWARD PLOT #
    ################
    # Create the overlay subplot for the forward data, in a small plot at the top left of the figure
    ax_sub = fig.add_axes(sum_pos_1)
    fill_in_mean_subplot(
        forward_hist_pdf,
        zero_bin_forward_value,
        clamped_forward_prob_mass,
        ax_sub,
        color="b",
        data_name="Forward Activation",
    )

    # Create the same for the gradients
    ax_2_sub = fig.add_axes(sum_pos_2)
    fill_in_mean_subplot(
        grad_hist_pdf,
        zero_bin_grad_value,
        clamped_grad_prob_mass,
        ax_2_sub,
        color="g",
        data_name="Gradients",
    )

    # Save fig with high resolution
    # NOTE: dpi = dots per inch, where a smaller value means faster plot generation but less resolution
    fig.savefig(filename, dpi=450)
    plt.close()
