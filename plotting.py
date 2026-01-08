import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv


def plot_layer_usage_heatmap(contrib):
    """
    contrib: numpy array [25, n_layers]
    """
    plt.figure(figsize=(8, 5))

    im = plt.imshow(
        contrib,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(im, label="Probe weight L2 norm")

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Grid cell (flattened index)", fontsize=12)

    plt.title(
        "Layer usage by linear probe for goal-grid reconstruction",
        fontsize=13,
        pad=8,
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()



def plot_metric_vs_layer(metr, name):
    """
    metr: 1D array-like of shape [n_layers]
    CHATGPT function
    """
    layers = np.arange(len(metr))

    plt.figure(figsize=(6.0, 4.0))  # NeurIPS-friendly size

    plt.plot(
        layers,
        metr,
        linewidth=2.0,
        marker="o",
        markersize=4,
    )

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(name, fontsize=12)

    plt.title(
        f"{name} vs layer",
        fontsize=13,
        pad=8,
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.8,
        alpha=0.4,
    )

    plt.tight_layout()
    plt.show()

def generate_colormap(number_of_distinct_colors: int = 80):
    # Amazing code from: https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)

def plot_color_gradients(category, cmap_list):
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # This code is for visualisation of data
    # Create figure and adjust figure height to number of colormaps
    cmaps = {}

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(15, 5))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'Colormap Legend', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        for i in range(10):
            ax.text(0.03+i*0.1, 0.5, i, va='center', ha='right', fontsize=10, transform=ax.transAxes)
    

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list