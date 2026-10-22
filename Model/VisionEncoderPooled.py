import re
import open_clip
import torch
import torch.nn as nn
from contextlib import nullcontext
from peft import LoraConfig, get_peft_model


# ── Short aliases → open_clip model identifiers ──────────────────────────
# Usage:  --encoder_model clip       (OpenAI CLIP ViT-B/16, 768-dim)
#         --encoder_model bioclip    (BioCLIP  ViT-B/16,    768-dim)
#         --encoder_model bioclip2   (BioCLIP2 ViT-B/16,    768-dim)
# You can also pass a raw hub string like "hf-hub:imageomics/bioclip".
ENCODER_REGISTRY = {
    "clip":     ("ViT-B-16", "openai"),          # two-arg open_clip API
    "bioclip":  ("hf-hub:imageomics/bioclip",),   # single-arg hub string
    "bioclip2": ("hf-hub:imageomics/bioclip-2",), # single-arg hub string
}


def _load_clip_model(encoder_model_str: str):
    """
    Resolve encoder_model to an open_clip model + transforms.
    Accepts short aliases ('clip', 'bioclip', 'bioclip2') or raw hub strings.
    Returns (clip_model, preprocess, encoder_name_str).
    """
    key = encoder_model_str.strip().lower()
    entry = ENCODER_REGISTRY.get(key)

    if entry is not None:
        if len(entry) == 2:
            # Standard CLIP: needs (model_name, pretrained)
            model, _, preprocess = open_clip.create_model_and_transforms(entry[0], pretrained=entry[1])
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(entry[0])
        name = key.upper()
    else:
        # Raw hub string pass-through (e.g. "hf-hub:imageomics/bioclip")
        model, _, preprocess = open_clip.create_model_and_transforms(encoder_model_str)
        name = encoder_model_str
    return model, preprocess, name


def _auto_detect_lora_targets(visual_encoder: nn.Module) -> str:
    """
    Scan the visual encoder's named modules and build a regex that matches
    all linear projections inside transformer/attention/MLP blocks.
    Works for both open_clip ViT (resblocks) and HF ViT (encoder.layers).
    """
    linear_names = set()
    for name, mod in visual_encoder.named_modules():
        if isinstance(mod, nn.Linear):
            linear_names.add(name)

    # open_clip ViT pattern: transformer.resblocks.N.attn.out_proj / mlp.c_fc / mlp.c_proj
    oc_pattern = r"transformer\.resblocks\.\d+\.(attn\.(out_proj|in_proj_weight)|mlp\.c_fc|mlp\.c_proj)"
    if any(re.match(oc_pattern, n) for n in linear_names):
        # Use the proven regex the repo already had (just attn.out_proj + mlp)
        return r"transformer\.resblocks\.\d+\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)"

    # Fallback: target every Linear layer inside any "block" / "layer" module
    return r".*\.(out_proj|fc1|fc2|q_proj|k_proj|v_proj)"


class VisionEncoder(nn.Module):
    """
    OpenCLIP Vision Encoder with LoRA.

    Accepts --encoder_model as:
      'clip'      → OpenAI CLIP ViT-B/16   (768-dim)
      'bioclip'   → BioCLIP  ViT-B/16      (768-dim)
      'bioclip2'  → BioCLIP2 ViT-B/16      (768-dim)
      or any raw open_clip / hub string.

    LoRA target modules are auto-detected from the model architecture.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        clip_model, self.preprocess, self.encoder_name = _load_clip_model(args.encoder_model)
        self.visual_encoder = clip_model.visual

        # Freeze base weights
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        # LoRA — target modules auto-detected
        self.use_lora = getattr(args, "use_lora", True)
        if self.use_lora:
            target_regex = _auto_detect_lora_targets(self.visual_encoder)
            config = LoraConfig(
                r=getattr(args, "lora_r", 16),
                lora_alpha=getattr(args, "lora_alpha", 32),
                lora_dropout=getattr(args, "lora_dropout", 0.05),
                target_modules=target_regex,
                bias="none",
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, config)
            print(f"[VisionEncoder] {self.encoder_name} — LoRA applied:")
            self.visual_encoder.print_trainable_parameters()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ctx = nullcontext() if self.use_lora else torch.no_grad()
        with ctx:
            features = self.visual_encoder.forward_intermediates(pixel_values)
            image_features = features["image_intermediates"][11]
            image_features = image_features.reshape(
                image_features.shape[0], image_features.shape[1], -1
            )
        return image_features