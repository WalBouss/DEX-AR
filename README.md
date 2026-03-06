# DEX-AR

### Explainability for Autoregressive Vision-Language Models

_[Walid Bousselham](http://walidbousselham.com/)<sup>1</sup>, [Angie Boggust](http://angieboggust.com/)<sup>2</sup>, [Hendrik Strobelt](http://hendrik.strobelt.com/)<sup>3,4</sup> and [Hilde Kuehne](https://hildekuehne.github.io/)<sup>1,3</sup>_

<sup>1</sup> Tuebingen AI Center & University of Tuebingen,
<sup>2</sup> MIT CSAIL,
<sup>3</sup> MIT-IBM Watson AI Lab,
<sup>4</sup> IBM Research.

DEX-AR is an explainability method for autoregressive Vision-Language Models (VLMs) such as LLaVA. It produces spatial heatmaps that highlight which image regions the model relies on when generating each token of its response.

The method works by:
1. Computing layer-wise gradients of intermediate logits w.r.t. attention maps (logit lens).
2. Applying dynamic head filtering — weighting each attention head by how much it focuses on visual vs. textual tokens.
3. Aggregating per-token heatmaps into a sentence-level explainability map using visual relevance weights.

The following is the code for a wrapper around [HuggingFace Transformers](https://github.com/huggingface/transformers) LLaVA models to equip them with DEX-AR.


## :hammer: Installation

`dexar_torch` can be installed via pip:
```bash
pip install -e .
```

Or directly:
```bash
pip install dexar_torch
```

## :rocket: Demo

Run [`playground.py`](./playground.py) for a usage example:
```bash
python playground.py
```

## :computer: Usage

### Basic Example

```python
from PIL import Image
from dexar import DexarWrapper, visualize

# --- Load model ---
model = DexarWrapper.from_pretrained("llava-hf/llava-1.5-7b-hf", device="cuda")

# --- Load image ---
image = Image.open("./assets/cat_and_dog.jpg").convert("RGB").resize(336, 336)

# --- Compute DEX-AR ---
result = model.compute_dexar(
    image=image,
    target_sentence="The image features a dog and a cat sitting together in a grassy field",
    prompt="USER: <image>\nDescribe the image. ASSISTANT:",
)

# --- Visualize sentence-level heatmap ---
visualize(image=image, heatmap=result.sentence_heatmap, title="Sentence heatmap")
```

### Outputs

`compute_dexar` returns a `DexarResult` with:

| Attribute | Shape | Description |
|---|---|---|
| `per_token_heatmaps` | `[T, H, W]` | Filtered per-token heatmaps (head filtering applied, Eq. 5) |
| `per_token_heatmaps_unfiltered` | `[T, H, W]` | Unfiltered per-token heatmaps (plain sum over heads/layers) |
| `token_weights` | `[T]` | Visual relevance weights per token (Eq. 6) |
| `sentence_heatmap` | `[H, W]` | Filtered sentence-level heatmap (Eq. 6) |
| `sentence_heatmap_unfiltered` | `[H, W]` | Unfiltered sentence-level heatmap |
| `tokens` | `list[str]` | Decoded token strings |

### Per-Token Visualization

```python
from dexar import visualize_multi
import torch.nn.functional as F

# Upscale heatmaps for visualization
per_token = F.interpolate(
    result.per_token_heatmaps[None], scale_factor=14, mode="bilinear"
)[0]
visualize_multi(image=image, heatmaps=per_token, tokens=result.tokens)
```

### Supported Models

| Model | HuggingFace ID |
|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |
| BakLLaVA | `llava-hf/bakLlava-v1-hf` |

### Parameters

- **`layer_index`** (default: `-10`): Starting depth for gradient computation. Negative values count from the last layer.
- **`prompt`**: Prompt template with `<image>` placeholder for the visual input.

## :star: Acknowledgement

This code is built as a wrapper around [HuggingFace Transformers](https://github.com/huggingface/transformers). This project also takes inspiration from [LeGrad](https://github.com/WalBouss/LeGrad).

## :books: Citation

If you find this repository useful, please consider citing our work :pencil: and giving a star :star2: :
```
@article{bousselham2026dexar,
  author    = {Bousselham, Walid and Boggust, Angie and Strobelt, Hendrik and Kuehne, Hilde},
  title     = {DEX-AR: A Dynamic Explainability Method for Autoregressive Vision-Language Models},
  year      = {2026},
}
```
