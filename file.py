# !wget "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" -O /kaggle/working/CUB_200_2011.tgz

# # 2. Extract it into /kaggle/working/
# !tar -xzf /kaggle/working/CUB_200_2011.tgz -C /kaggle/working/

# print("Dataset downloaded and extracted successfully to /kaggle/working/CUB_200_2011")
# !pip install -q -U "transformers>=4.51.0" peft
import transformers, peft
print(transformers.__version__, peft.__version__)
# !pip install --upgrade "torchao>0.16.0"
# !sed -i 's/img_key=None, label_key=None/img_key="imagePath", label_key=None/' /kaggle/working/train_vlm.py
# !grep -n 'img_key="imagePath"' /kaggle/working/train_vlm.py
# # !ls /kaggle/working/CUB_200_2011/images | head -3
%%writefile /kaggle/working/train_vlm.py
# ===== CLIP vision (LoRA) + Qwen3 decoder (LoRA), DDP on 2xT4, CUB-200 | name-weighted loss + light description loss =====
import os, sys, json, math, random, time, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from contextlib import nullcontext
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision import transforms as T
from transformers import (CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

# =========================== PATHS ===========================
TRAIN_JSON = "/kaggle/input/datasets/neurallogica/dataset/train.json"
TEST_JSON  = "/kaggle/input/datasets/neurallogica/dataset/test.json"
IMAGES_DIR = "/kaggle/working/CUB_200_2011/images"
OUT_DIR    = "/kaggle/working"
# =============================================================

CFG = dict(
    img_key="imagePath", desc_key="gt",      # label comes from the class folder name, NOT from gt
    clip="openai/clip-vit-base-patch16",     # "openai/clip-vit-large-patch14" is stronger if memory allows
    llm="Qwen/Qwen3-1.7B",                   # "Qwen/Qwen3-4B" for more capacity (use batch_size=8)
    epochs=12, eval_every=4,
    batch_size=16, grad_accum=1,             # PER GPU -> effective batch = 16 * 2 GPUs * 1 = 32
    lr_clip_lora=1e-4, lr_llm_lora=2e-4, lr_proj=1e-3,
    weight_decay=0.01, warmup_frac=0.05, label_smoothing=0.05, grad_clip=1.0,
    name_weight=1.0,                         # loss on the species-name tokens (main objective)
    desc_weight=0.15,                        # loss on the description tokens (light auxiliary); 0 disables it
    desc_max_tokens=64,
    aux_weight=0.5,                          # auxiliary classification loss on CLIP pooled feature
    lora_r_clip=16, lora_r_llm=32, lora_alpha_mult=2, lora_dropout=0.1,
    grad_ckpt=True, workers=2, eval_train_subset=2000, seed=42, out_dir=OUT_DIR,
)

# ------------------------------- DDP setup -------------------------------
RANK = int(os.environ.get("RANK", 0)); WORLD = int(os.environ.get("WORLD_SIZE", 1)); LOCAL = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(LOCAL); dev = torch.device("cuda", LOCAL)
if WORLD > 1: dist.init_process_group("nccl")
MAIN = RANK == 0
def log(*a):
    if MAIN: print(*a, flush=True)
def barrier():
    if WORLD > 1: dist.barrier()
def reduce(vals):
    t = torch.tensor(vals, dtype=torch.float64, device=dev)
    if WORLD > 1: dist.all_reduce(t)
    return t.tolist()
random.seed(CFG["seed"] + RANK); np.random.seed(CFG["seed"] + RANK); torch.manual_seed(CFG["seed"] + RANK)
log(f"GPUs visible: {torch.cuda.device_count()} | world_size={WORLD}")
CKPT_DIR = f'{CFG["out_dir"]}/checkpoints'; BEST_DIR = f'{CFG["out_dir"]}/best_model'
if MAIN: os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(BEST_DIR, exist_ok=True)

# ------------------------------- DATA (json -> paths + labels + description) -------------------------------
assert os.path.isdir(IMAGES_DIR), f"IMAGES_DIR not found: {IMAGES_DIR}"
class_dirs = sorted(d for d in os.listdir(IMAGES_DIR) if os.path.isdir(f"{IMAGES_DIR}/{d}"))
idx1, idx2 = {}, {}
for d in class_dirs:
    for f in os.listdir(f"{IMAGES_DIR}/{d}"):
        idx1[f] = f"{IMAGES_DIR}/{d}/{f}"; idx2[f"{d}/{f}"] = f"{IMAGES_DIR}/{d}/{f}"

def norm(s):
    s = re.sub(r"^\d+\.", "", str(s).strip())
    return re.sub(r"\s+", " ", s.replace("_", " ").replace("-", " ")).lower().strip()
class_names = [norm(d) for d in class_dirs]
name2idx = {n: i for i, n in enumerate(class_names)}

def load_json(p):
    txt = open(p).read()
    try: d = json.loads(txt)
    except json.JSONDecodeError: return [json.loads(l) for l in txt.splitlines() if l.strip()]
    if isinstance(d, dict):
        for k in ("data", "images", "annotations", "items", "samples", "train", "test"):
            if isinstance(d.get(k), list): return d[k]
        return [{"imagePath": k, "gt": v} for k, v in d.items()]
    return d

def resolve(p):
    p = str(p).replace("\\", "/")
    for c in (p, f"{IMAGES_DIR}/{p}", f"{os.path.dirname(IMAGES_DIR)}/{p}"):
        if os.path.isfile(c): return c
    return idx2.get("/".join(p.split("/")[-2:])) or idx1.get(os.path.basename(p))

def clean_desc(d):
    d = re.sub(r'^\s*It is a species of\s*"[^"]*"\s*as it has\s*', "", str(d or "")).strip()
    return d

def parse(path):
    recs = load_json(path); out, miss = [], 0
    log(f"[{os.path.basename(path)}] {len(recs)} records | first record: {recs[0]}")
    for r in recs:
        ip = r.get(CFG["img_key"]) if isinstance(r, dict) else (r[0] if isinstance(r, (list, tuple)) else r)
        desc = clean_desc(r.get(CFG["desc_key"])) if isinstance(r, dict) else ""
        fp = resolve(ip) if ip is not None else None
        if fp is None: miss += 1; continue
        out.append((fp, os.path.basename(os.path.dirname(fp)), desc))     # label = class folder name
    log(f"   usable={len(out)} missing_images={miss}")
    assert out, "No images resolved - check IMAGES_DIR and img_key in CFG"
    return out
tr_raw, te_raw = parse(TRAIN_JSON), parse(TEST_JSON)

for _, lb, _d in tr_raw + te_raw:                  # unseen folder names -> new classes
    if norm(lb) not in name2idx:
        name2idx[norm(lb)] = len(class_names); class_names.append(norm(lb))
tr_p = [x[0] for x in tr_raw]; tr_y = np.array([name2idx[norm(x[1])] for x in tr_raw]); tr_d = [x[2] for x in tr_raw]
te_p = [x[0] for x in te_raw]; te_y = np.array([name2idx[norm(x[1])] for x in te_raw])
NCLS = len(class_names)
train_classes = sorted(set(tr_y.tolist()))
log(f"classes={NCLS} (in train: {len(train_classes)}) train={len(tr_p)} test={len(te_p)}")

MEAN, STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)
tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=7),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
tf_eval = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize(MEAN, STD)])
class ImgDS(Dataset):                              # returns (image, class, record index)
    def __init__(self, p, y, tf): self.p, self.y, self.tf = p, y, tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i]), i
def collate(b): return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b]), torch.tensor([x[2] for x in b])
W = CFG["workers"]
def mk(ds, bs, sampler=None, shuffle=False):
    return DataLoader(ds, batch_size=bs, sampler=sampler, shuffle=shuffle, num_workers=W, collate_fn=collate,
                      pin_memory=True, persistent_workers=True, drop_last=sampler is not None)
train_ds = ImgDS(tr_p, tr_y, tf_train)
sampler = DistributedSampler(train_ds, num_replicas=WORLD, rank=RANK, shuffle=True, drop_last=True, seed=CFG["seed"])
train_dl = mk(train_ds, CFG["batch_size"], sampler)
def shard(ds): return Subset(ds, list(range(RANK, len(ds), WORLD)))   # exact (no padding) eval sharding
n_sub = min(CFG["eval_train_subset"] or len(tr_p), len(tr_p))
train_eval_dl = mk(shard(ImgDS(tr_p[:n_sub], tr_y[:n_sub], tf_eval)), 32)
test_dl = mk(shard(ImgDS(te_p, te_y, tf_eval)), 32)

# ------------------------------- TOKENS / TRIE -------------------------------
if RANK != 0: barrier()                       # rank 0 downloads weights first, the others read the cache
tok = AutoTokenizer.from_pretrained(CFG["llm"])
EOS = tok.eos_token_id
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
SEP = tok("\nDescription:", add_special_tokens=False).input_ids
name_key = [tok(" " + n, add_special_tokens=False).input_ids + SEP for n in class_names]   # name tokens + separator
MAX_NEW = max(len(a) for a in name_key) + 2
desc_tok = [(tok(" " + d, add_special_tokens=False).input_ids[:CFG["desc_max_tokens"]] + [EOS]) if d else [EOS] for d in tr_d]
PROMPT = tok("Question: Which bird species is in this image?\nAnswer:", return_tensors="pt", add_special_tokens=False).input_ids
TRIE, LOOKUP = {}, {}
for c in train_classes:                       # constrained decoding: output is always a valid class name
    node = TRIE
    for t in name_key[c]: node = node.setdefault(t, {})
    LOOKUP[tuple(name_key[c])] = c
def allowed_fn(batch_id, input_ids):
    node = TRIE
    for t in input_ids.tolist():
        node = node.get(t)
        if node is None: return [EOS]
    return list(node.keys()) or [EOS]

# ------------------------------- MODEL -------------------------------
def lora_cfg(r, targets):
    return LoraConfig(r=r, lora_alpha=r * CFG["lora_alpha_mult"], lora_dropout=CFG["lora_dropout"],
                      target_modules=targets, bias="none")
class VLM(nn.Module):
    def __init__(self, ncls):
        super().__init__()
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
        self.register_buffer("prompt_ids", PROMPT, persistent=False)

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
        pre, _ = self.prefix(px)
        out = self.llm.generate(inputs_embeds=pre, attention_mask=torch.ones(pre.shape[:2], dtype=torch.long, device=pre.device),
                                max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, eos_token_id=EOS, pad_token_id=PAD,
                                prefix_allowed_tokens_fn=allowed_fn)
        res = []
        for s in out.tolist():
            s = s[:s.index(EOS)] if EOS in s else s
            res.append(LOOKUP.get(tuple(s), -1))
        return res

model = VLM(NCLS).to(dev)
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()          # fp32 trainable weights (LoRA/proj/head) for GradScaler
if RANK == 0: barrier()
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
log(f"trainable params: {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M ({100*n_tr/n_all:.2f}%)")
ddp = DDP(model, device_ids=[LOCAL], broadcast_buffers=False) if WORLD > 1 else model

def make_ans(y, descs=None):
    """target = name tokens (+ description tokens when descs given). Returns ids, attn mask, name mask, desc mask."""
    seqs, nl, dl = [], [], []
    for j, c in enumerate(y.tolist()):
        n = name_key[c]; d = descs[j] if descs is not None else []
        seqs.append(n + d); nl.append(len(n)); dl.append(len(d))
    L = max(map(len, seqs)); Bn = len(seqs)
    a = torch.full((Bn, L), PAD, dtype=torch.long)
    m, nm, dm = (torch.zeros(Bn, L, dtype=torch.long) for _ in range(3))
    for i, s in enumerate(seqs):
        a[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1; nm[i, :nl[i]] = 1; dm[i, nl[i]:nl[i] + dl[i]] = 1
    return a.to(dev), m.to(dev), nm.to(dev).bool().reshape(-1), dm.to(dev).bool().reshape(-1)

# ------------------------------- EVAL (name only, sharded across both GPUs) -------------------------------
@torch.no_grad()
def eval_loss(dl):
    model.eval(); s = [0.0, 0.0, 0.0, 0.0]               # ce_sum, tokens, head_correct, n
    for px, y, _ in dl:
        px = px.to(dev); a, m, nm, dm = make_ans(y)
        with torch.autocast("cuda", dtype=torch.float16): lg, tgt, aux = model(px, a, m)
        s[0] += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum").item()
        s[1] += (tgt != -100).sum().item(); s[2] += (aux.argmax(1).cpu() == y).sum().item(); s[3] += len(y)
    s = reduce(s); return s[0] / s[1], s[2] / s[3]

@torch.no_grad()
def eval_acc(dl):
    model.eval(); c = n = 0
    for px, y, _ in tqdm(dl, desc="accuracy", leave=False, disable=not MAIN):
        with torch.autocast("cuda", dtype=torch.float16): pr = model.predict(px.to(dev))
        c += sum(p == t for p, t in zip(pr, y.tolist())); n += len(y)
    c, n = reduce([c, n]); return c / n

# ------------------------------- OPTIMIZER -------------------------------
groups = {"clip": [], "llm": [], "proj": []}
for n, p in model.named_parameters():
    if p.requires_grad:
        groups["clip" if n.startswith("vision.") else "proj" if n.startswith(("proj.", "head.")) else "llm"].append(p)
opt = torch.optim.AdamW([{"params": groups["clip"], "lr": CFG["lr_clip_lora"]},
                         {"params": groups["llm"], "lr": CFG["lr_llm_lora"]},
                         {"params": groups["proj"], "lr": CFG["lr_proj"]}], weight_decay=CFG["weight_decay"])
steps_per_epoch = math.ceil(len(train_dl) / CFG["grad_accum"]); total_steps = steps_per_epoch * CFG["epochs"]
sched = get_cosine_schedule_with_warmup(opt, int(CFG["warmup_frac"] * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda")
all_trainable = [p for g in groups.values() for p in g]

# ------------------------------- CKPT / PLOT (rank 0 only) -------------------------------
def save_ckpt(path, epoch, hist):
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"epoch": epoch, "trainable_state": state, "history": hist, "class_names": class_names, "cfg": CFG}, path)
def plot(hist):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(hist["epoch"], hist["train_loss"], "o-", label="train name loss"); ax[0].plot(hist["epoch"], hist["test_loss"], "s-", label="test name loss")
    ax[0].plot(hist["epoch"], hist["desc_loss"], "^--", alpha=.5, label="train desc loss")
    ax[0].set(xlabel="epoch", ylabel="loss (CE / token)", title="Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist["epoch"], hist["head_acc"], "^--", alpha=.6, label="test aux-head acc")
    if hist["acc_epoch"]:
        ax[1].plot(hist["acc_epoch"], hist["train_acc"], "o-", label="train acc"); ax[1].plot(hist["acc_epoch"], hist["test_acc"], "s-", label="test acc")
        for e, a in zip(hist["acc_epoch"], hist["test_acc"]): ax[1].annotate(f"{a:.3f}", (e, a), textcoords="offset points", xytext=(0, 6))
    ax[1].set(xlabel="epoch", ylabel="accuracy", title=f"Accuracy (generation every {CFG['eval_every']} epochs)", xlim=(0, CFG["epochs"] + .5), ylim=(0, 1.05))
    ax[1].legend(); ax[1].grid(alpha=.3); plt.tight_layout(); plt.savefig(f'{CFG["out_dir"]}/loss_curves.png', dpi=130); plt.close()

# ------------------------------- TRAIN -------------------------------
hist = dict(epoch=[], train_loss=[], test_loss=[], desc_loss=[], head_acc=[], acc_epoch=[], train_acc=[], test_acc=[])
best_acc, t0, A = -1.0, time.time(), CFG["grad_accum"]
LS, DW, NW, AW = CFG["label_smoothing"], CFG["desc_weight"], CFG["name_weight"], CFG["aux_weight"]
for ep in range(1, CFG["epochs"] + 1):
    sampler.set_epoch(ep); ddp.train()
    run = torch.zeros(4, device=dev, dtype=torch.float64)       # name_ce_sum, name_tokens, desc_ce_sum, desc_tokens
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(train_dl, desc=f"epoch {ep}/{CFG['epochs']}", disable=not MAIN)
    for i, (px, y, ridx) in enumerate(pbar):
        step_now = (i + 1) % A == 0 or (i + 1) == len(train_dl)
        descs = [desc_tok[j] for j in ridx.tolist()] if DW > 0 else None
        px = px.to(dev, non_blocking=True); a, m, nm, dm = make_ans(y, descs); y = y.to(dev)
        with (nullcontext() if (step_now or WORLD == 1) else ddp.no_sync()):
            with torch.autocast("cuda", dtype=torch.float16): lg, tgt, aux = ddp(px, a, m)
            lg = lg.float().reshape(-1, lg.size(-1)); tgt = tgt.reshape(-1)
            per = F.cross_entropy(lg, tgt, ignore_index=-100, reduction="none", label_smoothing=LS)
            loss = NW * per[nm].mean() + AW * F.cross_entropy(aux.float(), y, label_smoothing=LS)
            if DW > 0 and dm.any(): loss = loss + DW * per[dm].mean()
            scaler.scale(loss / A).backward()
        with torch.no_grad():
            raw = F.cross_entropy(lg, tgt, ignore_index=-100, reduction="none")
            run[0] += raw[nm].sum().double(); run[1] += nm.sum(); run[2] += raw[dm].sum().double(); run[3] += dm.sum()
        if step_now:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(all_trainable, CFG["grad_clip"])
            prev_scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= prev_scale: sched.step()      # skip LR step if GradScaler skipped the update
            opt.zero_grad(set_to_none=True)
        if MAIN: pbar.set_postfix(name_loss=f"{(run[0]/run[1]).item():.4f}")

    r = reduce(run.tolist()); tr_loss = r[0] / r[1]; d_loss = r[2] / max(r[3], 1)
    te_loss, head_acc = eval_loss(test_dl)
    hist["epoch"].append(ep); hist["train_loss"].append(tr_loss); hist["test_loss"].append(te_loss)
    hist["desc_loss"].append(d_loss); hist["head_acc"].append(head_acc)
    msg = f"[ep {ep}] name_loss train={tr_loss:.4f} test={te_loss:.4f} gap={te_loss-tr_loss:+.4f} | desc_loss={d_loss:.4f} | head_acc={head_acc:.4f}"
    improved = False
    if ep % CFG["eval_every"] == 0:
        tr_acc, te_acc = eval_acc(train_eval_dl), eval_acc(test_dl)
        hist["acc_epoch"].append(ep); hist["train_acc"].append(tr_acc); hist["test_acc"].append(te_acc)
        msg += f" | train_acc={tr_acc:.4f} test_acc={te_acc:.4f}"
        if te_acc > best_acc: best_acc, improved = te_acc, True; msg += "  <-- new best"
    log(msg)
    if MAIN:
        if improved:
            save_ckpt(f"{BEST_DIR}/best.pt", ep, hist)
            model.vision.save_pretrained(f"{BEST_DIR}/clip_lora"); model.llm.save_pretrained(f"{BEST_DIR}/qwen3_lora")
            torch.save({"proj": model.proj.state_dict(), "head": model.head.state_dict()}, f"{BEST_DIR}/projector_head.pt")
        save_ckpt(f"{CKPT_DIR}/epoch_{ep:02d}.pt", ep, hist)
        json.dump(hist, open(f'{CFG["out_dir"]}/history.json', "w"), indent=2); plot(hist)
    barrier()
log(f"\nDone in {(time.time()-t0)/60:.1f} min | best test acc = {best_acc:.4f} (saved in {BEST_DIR})")
if WORLD > 1: dist.destroy_process_group()
