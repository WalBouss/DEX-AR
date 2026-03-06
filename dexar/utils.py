import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, List


def min_max(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalization."""
    return (x - x.min()) / (x.max() - x.min())


def topk_norm(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Top-k norm scoring: average of top-k absolute values along dim."""
    return torch.topk(torch.abs(x), k=k, dim=dim)[0].sum(dim=dim) / k


def visualize(
    image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float = 0.6,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Overlay a heatmap on a PIL image.

    Args:
        image: PIL Image to overlay on.
        heatmap: 2D tensor [H, W] with values in [0, 1].
        alpha: Blending factor for the heatmap overlay.
        title: Optional title for the plot.
        save_path: If provided, saves the figure to this path.
    """
    H, W = heatmap.shape[-2:]
    image_resized = image.resize((W, H))

    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_uint8 = (heatmap_np * 255).astype("uint8")

    img_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    heat_map_cv = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    viz = (1 - alpha) * img_cv + alpha * heat_map_cv
    viz = cv2.cvtColor(viz.astype("uint8"), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(viz)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close(fig)


def visualize_multi(
    image: Image.Image,
    heatmaps: torch.Tensor,
    tokens: Optional[List[str]] = None,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> None:
    """Overlay multiple per-token heatmaps on a PIL image.

    Args:
        image: PIL Image to overlay on.
        heatmaps: 3D tensor [num_tokens, H, W] with values in [0, 1].
        tokens: Optional list of token strings for titles.
        alpha: Blending factor for the heatmap overlay.
        save_path: If provided, saves the figure to this path prefix.
    """
    num_tokens = heatmaps.shape[0]
    H, W = heatmaps.shape[-2:]
    image_resized = image.resize((W, H))
    img_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)

    if tokens is None:
        tokens = [str(i) for i in range(num_tokens)]

    heatmaps_np = heatmaps.detach().cpu().numpy()

    fig, axes = plt.subplots(1, num_tokens + 1, figsize=(3 * (num_tokens + 1), 3))
    axes[0].imshow(image_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    overlays = []
    for i in range(num_tokens):
        heatmap_uint8 = (heatmaps_np[i] * 255).astype("uint8")
        heat_map_cv = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        viz = (1 - alpha) * img_cv + alpha * heat_map_cv
        viz = cv2.cvtColor(viz.astype("uint8"), cv2.COLOR_BGR2RGB)
        overlays.append(viz)
        axes[i + 1].imshow(viz)
        axes[i + 1].set_title(tokens[i])
        axes[i + 1].axis("off")

    fig.tight_layout()
    if save_path is not None:
        stem, ext = os.path.splitext(save_path)
        ext = ext or ".png"
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        for i in range(num_tokens):
            token_safe = tokens[i].strip().replace(" ", "_").replace("/", "_")
            individual_path = f"{stem}_{token_safe}{ext}"
            fig_i, ax_i = plt.subplots(figsize=(3, 3))
            ax_i.imshow(overlays[i])
            ax_i.set_title(tokens[i])
            ax_i.axis("off")
            fig_i.tight_layout()
            fig_i.savefig(individual_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig_i)
    plt.show()
    plt.close(fig)
