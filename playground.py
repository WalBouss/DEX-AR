"""DEX-AR usage example.

Produces per-token heatmaps and a sentence-level explainability map
for a given image and target sentence.

Usage:
    python playground.py
"""

import os

from PIL import Image
import torch.nn.functional as F
from dexar import DexarWrapper, visualize, visualize_multi


def main():
    filtered_dir = "./filtered"
    unfiltered_dir = "./unfiltered"
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(unfiltered_dir, exist_ok=True)

    model_name = "llava-hf/llava-1.5-7b-hf"
    # model_name = "llava-hf/bakLlava-v1-hf"
    device = "cuda"
    layer_index = 0
    # --- Load model ---
    model = DexarWrapper.from_pretrained(
        model_name, device=device, layer_index=layer_index
    )

    # --- Load a sample image ---
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    path_to_image = "./assets/cat_and_dog.jpg"
    image = Image.open(path_to_image).convert("RGB").resize((336, 336))

    # --- Compute DEX-AR ---
    result = model.compute_dexar(
        image=image,
        target_sentence="The image features a dog and a cat sitting together in a grassy field",
        prompt="USER: <image>\nDescribe the image. ASSISTANT:",
    )

    # --- Print token info ---
    print("Tokens:", result.tokens)
    print("Token weights (delta^t):", result.token_weights)
    print("Per-token heatmaps shape:", result.per_token_heatmaps.shape)
    print("Sentence heatmap shape:", result.sentence_heatmap.shape)

    # --- Filtered: sentence-level heatmap (with head filtering, Eq. 5) ---
    sentence_heatmap = F.interpolate(
        result.sentence_heatmap[None, None], scale_factor=14, mode="bilinear"
    )[0, 0]
    visualize(
        image=image, heatmap=sentence_heatmap,
        title="Sentence heatmap (filtered)",
        save_path=os.path.join(filtered_dir, "sentence_heatmap.png"),
    )

    # --- Filtered: per-token heatmaps ---
    per_token_heatmaps = F.interpolate(
        result.per_token_heatmaps[None], scale_factor=14, mode="bilinear"
    )[0]
    visualize_multi(
        image=image, heatmaps=per_token_heatmaps, tokens=result.tokens,
        save_path=os.path.join(filtered_dir, "per_token_heatmaps.png"),
    )

    # --- Unfiltered: sentence-level heatmap (no head filtering) ---
    sentence_heatmap_unfiltered = F.interpolate(
        result.sentence_heatmap_unfiltered[None, None], scale_factor=14, mode="bilinear"
    )[0, 0]
    visualize(
        image=image, heatmap=sentence_heatmap_unfiltered,
        title="Sentence heatmap (unfiltered)",
        save_path=os.path.join(unfiltered_dir, "sentence_heatmap.png"),
    )

    # --- Unfiltered: per-token heatmaps ---
    per_token_heatmaps_unfiltered = F.interpolate(
        result.per_token_heatmaps_unfiltered[None], scale_factor=14, mode="bilinear"
    )[0]
    visualize_multi(
        image=image, heatmaps=per_token_heatmaps_unfiltered, tokens=result.tokens,
        save_path=os.path.join(unfiltered_dir, "per_token_heatmaps.png"),
    )


if __name__ == "__main__":
    main()
