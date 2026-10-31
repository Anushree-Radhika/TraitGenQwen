import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from config import CFG


# LoRA is applied to these linear layers
CLIP_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LLM_LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_cfg(rank, targets):
    """Build a LoRA config. alpha = rank * lora_alpha_mult."""

    return LoraConfig(
        r=rank,
        lora_alpha=rank * CFG["lora_alpha_mult"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=targets,
        bias="none",
    )


class VLM(nn.Module):
    """CLIP vision encoder (LoRA) -> projector -> Qwen3 decoder (LoRA).

    Pieces:
        self.vision  : CLIP ViT with LoRA
        self.proj    : MLP mapping CLIP features into the LLM embedding space
        self.llm     : Qwen3 with LoRA
        self.head    : auxiliary classifier on CLIP's pooled feature
    """

    # =========================================================================
    # BUILD
    # =========================================================================

    def __init__(self, ncls, codec):
        super().__init__()
        self.codec = codec

        # ---------- Load frozen base models in fp16 ----------
        clip = CLIPVisionModel.from_pretrained(CFG["clip"], torch_dtype=torch.float16)
        llm  = AutoModelForCausalLM.from_pretrained(CFG["llm"], torch_dtype=torch.float16)

        # ---------- Wrap both with LoRA ----------
        self.vision = get_peft_model(clip, lora_cfg(CFG["lora_r_clip"], CLIP_LORA_TARGETS))
        self.llm    = get_peft_model(llm,  lora_cfg(CFG["lora_r_llm"],  LLM_LORA_TARGETS))

        # ---------- Gradient checkpointing (saves memory, costs some speed) ----------
        if CFG["grad_ckpt"]:
            ckpt_kwargs = dict(gradient_checkpointing_kwargs={"use_reentrant": False})

            clip.gradient_checkpointing_enable(**ckpt_kwargs)
            llm.gradient_checkpointing_enable(**ckpt_kwargs)

        # ---------- Projector and auxiliary head ----------
        vision_dim = clip.config.hidden_size
        llm_dim    = llm.config.hidden_size

        self.proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # Auxiliary classifier on the pooled CLIP feature
        self.head = nn.Linear(vision_dim, ncls)

        # Fixed text prompt (stored as a buffer so it moves with the model, not saved)
        self.register_buffer("prompt_ids", codec.prompt, persistent=False)

    # =========================================================================
    # IMAGE -> VISUAL TOKENS
    # =========================================================================

    def encode_image(self, px):
        """Turn images into visual tokens for the LLM.

        Shapes (for ViT-B/16 at 224px):
            CLIP output        : (B, 197, Dv)  = 1 CLS + 14x14 patches
            after 2x2 pooling  : (B,  50, Dv)  = 1 CLS + 7x7 patches
            after projector    : (B,  50, Dl)

        Returns:
            visual_tokens : (B, 50, Dl)
            aux_logits    : (B, ncls)
        """

        out    = self.vision(pixel_values=px)
        hidden = out.last_hidden_state

        # Split CLS token from the patch tokens
        cls_token = hidden[:, :1]
        patches   = hidden[:, 1:]

        # Reshape patches to a square grid and average-pool 2x2 -> fewer tokens
        B, N, C = patches.shape
        grid    = int(N ** 0.5)

        patches = patches.transpose(1, 2).reshape(B, C, grid, grid)
        patches = F.avg_pool2d(patches, 2)
        patches = patches.flatten(2).transpose(1, 2)

        # Project into the LLM embedding space
        visual_tokens = self.proj(torch.cat([cls_token, patches], dim=1))

        # Auxiliary classification from the pooled CLIP feature
        aux_logits = self.head(out.pooler_output.float())

        return visual_tokens, aux_logits

    def prefix(self, px):
        """Build the LLM input prefix: [visual tokens | prompt tokens]."""

        visual_tokens, aux_logits = self.encode_image(px)

        prompt_emb = self.llm.get_input_embeddings()(self.prompt_ids)
        prompt_emb = prompt_emb.expand(visual_tokens.size(0), -1, -1)

        prefix = torch.cat([visual_tokens.to(prompt_emb.dtype), prompt_emb], dim=1)

        return prefix, aux_logits

    # =========================================================================
    # TRAINING FORWARD PASS
    # =========================================================================

    def forward(self, px, ans, ans_mask):
        """Teacher-forced forward pass.

        Args:
            px       : images                      (B, 3, 224, 224)
            ans      : target token ids            (B, L)
            ans_mask : 1 for real tokens, 0 = pad  (B, L)

        Returns:
            logits     : predictions for the answer tokens   (B, L, vocab)
            targets    : ans with padding set to -100        (B, L)
            aux_logits : auxiliary classifier output         (B, ncls)
        """

        prefix, aux_logits = self.prefix(px)
        B, L = ans.shape

        # ---- Input = [prefix | answer embeddings] ----
        ans_emb = self.llm.get_input_embeddings()(ans)
        x = torch.cat([prefix, ans_emb], dim=1)

        # ---- Attention mask: prefix is always visible, then the answer mask ----
        prefix_mask = torch.ones(B, prefix.size(1), device=ans.device, dtype=ans_mask.dtype)
        mask = torch.cat([prefix_mask, ans_mask], dim=1)

        # ---- Run the decoder body, then the LM head on the answer positions only ----
        inner = self.llm.base_model.model
        hidden = inner.model(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state

        # Position t predicts token t+1, so shift by one
        logits = inner.lm_head(hidden[:, -L - 1:-1])

        targets = ans.masked_fill(ans_mask == 0, -100)

        return logits, targets, aux_logits

    # =========================================================================
    # INFERENCE
    # =========================================================================

    @torch.no_grad()
    def predict(self, px):
        """Generate a species name for each image using constrained decoding.

        Returns:
            list of class ids (-1 if the generated text is not a valid class name)
        """

        codec = self.codec
        prefix, _ = self.prefix(px)

        attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)

        generated = self.llm.generate(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            max_new_tokens=codec.max_new,
            do_sample=False,
            num_beams=1,
            eos_token_id=codec.eos,
            pad_token_id=codec.pad,
            prefix_allowed_tokens_fn=codec.allowed_fn,      # trie: only valid class names
        )

        # ---- Convert generated token ids back to class ids ----
        predictions = []

        for tokens in generated.tolist():

            # Cut at the first EOS
            if codec.eos in tokens:
                tokens = tokens[:tokens.index(codec.eos)]

            predictions.append(codec.lookup.get(tuple(tokens), -1))

        return predictions
