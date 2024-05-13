# The histogram plot range will extend negatively beyond the minimum quantization value by `HIST_XLIM_MIN` of the quantization range.
HIST_XMIN = 0.5
# The histogram plot range will extend positively beyond the maximum quantization value by `HIST_XMAS` of the quantization range.
HIST_XMAX = 0.5
# How many histogram bins per quantization bin
HIST_QUANT_BIN_RATIO = 5
# How many quantization bins to average in the smoothing average plot for the forward and grads
SMOOTH_WINDOW = 9

# Coordinates for the sub-plot for the forward histogram mini-plot
SUM_POS_1_DEFAULT = [0.18, 0.60, 0.1, 0.1]
# Coordinates for the sub-plot for the sensitivity analysis mini-plot
SUM_POS_2_DEFAULT = [0.75, 0.43, 0.1, 0.1]
