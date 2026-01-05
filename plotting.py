import matplotlib.pyplot as plt
import numpy as np

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