import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import os

matplotlib.use("Agg")

from ...settings import HIST_QUANT_BIN_RATIO, HIST_XMAX, HIST_XMIN


############
# SUBPLOTS #
############
def fill_in_mean_subplot(
    distribution: torch.Tensor,
    zero_bin_value: torch.Tensor,
    clamped_prob_mass: torch.Tensor,
    ax_sub: matplotlib.axes._axes.Axes,
    color: str = "blue",
    data_name: str = "",
):
    """
    Fills in the summary sub-plot. This involves calculating the mean intra-bin values, and plotting them.
    We also add a few interesting statistics:
    - the amount of not-on-bin-centroid probability mass
    - the zero-bin value
    - the amount of clamped probability mass

    Inputs:
    - distribution (torch.Tensor): the PDF we will be getting the mean intra-bin plot of.
    - zero_bin_value (torch.Tensor): the zero bin probability mass value.
    - clamped_prob_mass (torch.Tensor): the clamped probability mass scalar value.
    - ax_sub (matplotlib.axes._axes.Axes) = the Axes object we will manipulate to fill in the subplot.
    - color (str): the color of the imean intra-bin plot.
    - data_name (str): part of the subtitle, e.g. "Forward Activation", "Gradient", etc.
    """
    # Sum every HIST_QUANT_BIN_RATIO'th value in the histogram.
    intra_bin = torch.zeros(HIST_QUANT_BIN_RATIO)
    for step in torch.arange(HIST_QUANT_BIN_RATIO):
        intra_bin[step] = distribution[step::HIST_QUANT_BIN_RATIO].sum()
    indices = range(HIST_QUANT_BIN_RATIO)
    intra_bin = intra_bin.numpy()

    # Plot the intra-bin behavior as subplot
    ax_sub.bar(indices, intra_bin, color=color)

    # Remove tick labels and set background to transparent for the overlay subplot
    ax_sub.set_xticks(np.arange(0, HIST_QUANT_BIN_RATIO + 1, HIST_QUANT_BIN_RATIO / 2))
    ax_sub.set_xticklabels(
        [
            f"{int(i)}/{HIST_QUANT_BIN_RATIO}"
            for i in np.arange(0, HIST_QUANT_BIN_RATIO + 1, HIST_QUANT_BIN_RATIO / 2)
        ]
    )
    ax_sub.set_xlim(-0.5, HIST_QUANT_BIN_RATIO + 0.5)
    ax_sub.patch.set_alpha(1)

    # Add title (with summary-ish statistics) and labels
    title_str = f"{data_name}\nMean Intra-bin Behavior\n(Not-on-quant-bin-centroid\nprob mass: {intra_bin[1:].sum():.2f})\nZero-bin mass: {zero_bin_value:.2f}"
    title_str += f"\nClamped prob mass: {clamped_prob_mass:.6f}"
    ax_sub.set_title(title_str)
    ax_sub.set_ylabel("Prob")
    ax_sub.set_xlabel("Bins (0 and 1 are centroids)")
    ax_sub.axvline(x=0, color="black", linewidth=1)
    ax_sub.axvline(x=HIST_QUANT_BIN_RATIO, color="black", linewidth=1)


def draw_centroids_and_tensor_range(
    ax: matplotlib.axes._axes.Axes,
    bin_edges: torch.Tensor,
    qrange: int,
    tensor_min_index: torch.Tensor,
    tensor_max_index: torch.Tensor,
    scale: torch.Tensor,
):
    """
    Draws black vertical lines at each quantization centroid, and adds thick red lines at the edges
    of the floating point tensor, i.e. highlights its dynamic range.

    Inputs:
    - ax (matplotlib.axes._axes.Axes): the Axes object we will be manipulating to add the plot elements.
    - bin_edges (torch.Tensor): the histogram bin edges
    - qrange (int): the number of quantization bins
    - tensor_min_index (torch.Tensor): the minimum value in the floating point tensor.
    - tensor_max_index (torch.Tensor): the maximum value in the floating point tensor.
    - scale (torch.Tensor): the quantization scale.
    """
    # Draws black vertical lines
    for index, x_val in enumerate(
        np.arange(
            start=bin_edges[int(HIST_XMIN * qrange * HIST_QUANT_BIN_RATIO)],
            stop=bin_edges[-int(HIST_XMAX * qrange * HIST_QUANT_BIN_RATIO)],
            step=scale,
        )
    ):
        if index == 0:
            ax.axvline(
                x=x_val,
                color="black",
                linewidth=0.08,
                label="Quantization bin centroids",
            )
        else:
            ax.axvline(x=x_val, color="black", linewidth=0.08)

    # Draw vertical lines at dynamic range boundaries of forward tensor (1 quantization bin padding)
    ax.axvline(
        x=bin_edges[tensor_min_index] - scale,
        color="red",
        linewidth=1,
        label="Tensor dynamic range",
    )
    ax.axvline(x=bin_edges[tensor_max_index] + scale, color="red", linewidth=1)


###################
# DATA PROCESSING #
###################
def get_prob_mass_outside_quant_range(
    distribution: torch.Tensor, qrange: int
) -> torch.Tensor:
    """
    Returns the amount of probability mass outside the quantization range.
    """
    clamped_prob_mass = torch.sum(
        distribution[: int(HIST_XMIN * qrange * HIST_QUANT_BIN_RATIO)]
    ) + torch.sum(distribution[int((HIST_XMIN + 1) * qrange * HIST_QUANT_BIN_RATIO) :])
    return clamped_prob_mass


def moving_average(input_tensor, window_size):
    """
    Get a 1d moving average of a 1D torch tensor, used for creating a smoothed
    data distribution for the histograms.
    """
    # Create a 1D convolution kernel filled with ones
    kernel = torch.ones(1, 1, window_size) / window_size

    # Apply padding to handle boundary elements
    padding = (window_size - 1) // 2

    # Apply the convolution operation
    output_tensor = F.conv1d(
        input_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=padding
    )

    return output_tensor.squeeze()


###########
# PATHING #
###########


def create_double_level_plot_folder(file_path: str, lvl_1: str, lvl_2: str) -> str:
    weight_plot_folder = file_path / lvl_1 / lvl_2
    if not os.path.exists(weight_plot_folder):
        os.makedirs(weight_plot_folder, exist_ok=True)
    return weight_plot_folder
