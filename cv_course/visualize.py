import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List


def plot_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols > 1 else [axes]

    for i in range(n):
        axes[i].imshow(images[i], cmap="gray" if images[i].ndim == 2 else None)
        axes[i].axis("off")
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def plot_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    import matplotlib.patches as patches

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)

        label = ""
        if class_ids is not None and class_names is not None:
            label = class_names[class_ids[i]]
        if scores is not None:
            label += f" {scores[i]:.2f}"
        if label:
            ax.text(x1, y1 - 5, label, color="lime", fontsize=10)

    ax.axis("off")
    return ax


def plot_attention(
    attn_map: np.ndarray,
    image: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if image is not None:
        ax.imshow(image, alpha=0.6)
    ax.imshow(attn_map, cmap="jet", alpha=0.4)
    ax.axis("off")
    return ax
