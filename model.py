import torch, torch.nn as nn, torch.nn.functional as F
from transformers import CLIPVisionModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import CFG

def lora_cfg(r, targets):
    return LoraConfig(r=r, lora_alpha=r * CFG["lora_alpha_mult"], lora_dropout=CFG["lora_dropout"],
                      target_modules=targets, bias="none")

class VLM(nn.Module):
    def __init__(self, ncls, codec):
        super().__init__()
        self.codec = codec
        clip = CLIPVisionModel.from_pretrained(CFG["clip"], torch_dtype=torch.float16)
        llm = AutoModelForCausalLM.from_pretrained(CFG["llm"], torch_dtype=torch.float16)
        self.vision = get_peft_model(clip, lora_cfg(CFG["lora_r_clip"], ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        self.llm = get_peft_model(llm, lora_cfg(CFG["lora_r_llm"], ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        if CFG["grad_ckpt"]:
            ck = dict(gradient_checkpointing_kwargs={"use_reentrant": False})
            clip.gradient_checkpointing_enable(**ck); llm.gradient_checkpointing_enable(**ck)
        Dv, Dl = clip.config.hidden_size, llm.config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(Dv), nn.Linear(Dv, Dl), nn.GELU(), nn.Linear(Dl, Dl))
        self.head = nn.Linear(Dv, ncls)                                  # aux classifier on pooled CLIP feature
        self.register_buffer("prompt_ids", codec.prompt, persistent=False)

    def encode_image(self, px):
        o = self.vision(pixel_values=px)
        h = o.last_hidden_state
        cls, p = h[:, :1], h[:, 1:]
        B, N, C = p.shape; g = int(N ** 0.5)
        p = F.avg_pool2d(p.transpose(1, 2).reshape(B, C, g, g), 2).flatten(2).transpose(1, 2)
        return self.proj(torch.cat([cls, p], 1)), self.head(o.pooler_output.float())

    def prefix(self, px):
        v, aux = self.encode_image(px)
        pe = self.llm.get_input_embeddings()(self.prompt_ids).expand(v.size(0), -1, -1)
        return torch.cat([v.to(pe.dtype), pe], 1), aux

    def forward(self, px, ans, ans_mask):
        pre, aux = self.prefix(px); B, L = ans.shape
        x = torch.cat([pre, self.llm.get_input_embeddings()(ans)], 1)
        mask = torch.cat([torch.ones(B, pre.size(1), device=ans.device, dtype=ans_mask.dtype), ans_mask], 1)
        inner = self.llm.base_model.model
        hs = inner.model(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        logits = inner.lm_head(hs[:, -L - 1:-1])
        return logits, ans.masked_fill(ans_mask == 0, -100), aux

    @torch.no_grad()
    def predict(self, px):
        c = self.codec
        pre, _ = self.prefix(px)
        out = self.llm.generate(inputs_embeds=pre, attention_mask=torch.ones(pre.shape[:2], dtype=torch.long, device=pre.device),
                                max_new_tokens=c.max_new, do_sample=False, num_beams=1, eos_token_id=c.eos, pad_token_id=c.pad,
                                prefix_allowed_tokens_fn=c.allowed_fn)
        res = []
        for s in out.tolist():
            s = s[:s.index(c.eos)] if c.eos in s else s
            res.append(c.lookup.get(tuple(s), -1))
        return res
