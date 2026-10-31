import os, re, json
from dataclasses import dataclass
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision import transforms as T
import torch
from config import CFG, TRAIN_JSON, TEST_JSON, IMAGES_DIR, MEAN, STD
from dist_utils import RANK, WORLD, log

# ---------- name / json helpers ----------
def norm(s):
    s = re.sub(r"^\d+\.", "", str(s).strip())
    return re.sub(r"\s+", " ", s.replace("_", " ").replace("-", " ")).lower().strip()

def load_json(p):
    txt = open(p).read()
    try: d = json.loads(txt)
    except json.JSONDecodeError: return [json.loads(l) for l in txt.splitlines() if l.strip()]
    if isinstance(d, dict):
        for k in ("data", "images", "annotations", "items", "samples", "train", "test"):
            if isinstance(d.get(k), list): return d[k]
        return [{"imagePath": k, "gt": v} for k, v in d.items()]
    return d

def clean_desc(d):
    return re.sub(r'^\s*It is a species of\s*"[^"]*"\s*as it has\s*', "", str(d or "")).strip()

def index_images():
    assert os.path.isdir(IMAGES_DIR), f"IMAGES_DIR not found: {IMAGES_DIR}"
    class_dirs = sorted(d for d in os.listdir(IMAGES_DIR) if os.path.isdir(f"{IMAGES_DIR}/{d}"))
    idx1, idx2 = {}, {}
    for d in class_dirs:
        for f in os.listdir(f"{IMAGES_DIR}/{d}"):
            idx1[f] = f"{IMAGES_DIR}/{d}/{f}"; idx2[f"{d}/{f}"] = f"{IMAGES_DIR}/{d}/{f}"
    return class_dirs, idx1, idx2

def resolve(p, idx1, idx2):
    p = str(p).replace("\\", "/")
    for c in (p, f"{IMAGES_DIR}/{p}", f"{os.path.dirname(IMAGES_DIR)}/{p}"):
        if os.path.isfile(c): return c
    return idx2.get("/".join(p.split("/")[-2:])) or idx1.get(os.path.basename(p))

def parse(path, idx1, idx2):
    recs = load_json(path); out, miss = [], 0
    log(f"[{os.path.basename(path)}] {len(recs)} records | first record: {recs[0]}")
    for r in recs:
        ip = r.get(CFG["img_key"]) if isinstance(r, dict) else (r[0] if isinstance(r, (list, tuple)) else r)
        desc = clean_desc(r.get(CFG["desc_key"])) if isinstance(r, dict) else ""
        fp = resolve(ip, idx1, idx2) if ip is not None else None
        if fp is None:
            miss += 1; continue
        out.append((fp, os.path.basename(os.path.dirname(fp)), desc))
    log(f"   usable={len(out)} missing_images={miss}")
    assert out, "No images resolved - check IMAGES_DIR and img_key in CFG"
    return out

# ---------- torch datasets ----------
tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=7),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
tf_eval = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize(MEAN, STD)])

class ImgDS(Dataset):                              # returns (image, class, record index)
    def __init__(self, p, y, tf): self.p, self.y, self.tf = p, y, tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i]), i

def collate(b):
    return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b]), torch.tensor([x[2] for x in b])

def make_loader(ds, bs, sampler=None):
    return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=CFG["workers"], collate_fn=collate,
                      pin_memory=True, persistent_workers=True, drop_last=sampler is not None)

def shard(ds):                                     # exact (no padding) eval sharding
    return Subset(ds, list(range(RANK, len(ds), WORLD)))

# ---------- everything the rest of the code needs ----------
@dataclass
class DataBundle:
    class_names: list
    train_classes: list
    train_descs: list
    sampler: DistributedSampler
    train_dl: DataLoader
    train_eval_dl: DataLoader
    test_dl: DataLoader
    @property
    def ncls(self): return len(self.class_names)

def build_data():
    class_dirs, idx1, idx2 = index_images()
    class_names = [norm(d) for d in class_dirs]
    name2idx = {n: i for i, n in enumerate(class_names)}

    tr_raw, te_raw = parse(TRAIN_JSON, idx1, idx2), parse(TEST_JSON, idx1, idx2)
    for _, lb, _d in tr_raw + te_raw:              # unseen folder names -> new classes
        if norm(lb) not in name2idx:
            name2idx[norm(lb)] = len(class_names); class_names.append(norm(lb))

    tr_p = [x[0] for x in tr_raw]; tr_y = np.array([name2idx[norm(x[1])] for x in tr_raw]); tr_d = [x[2] for x in tr_raw]
    te_p = [x[0] for x in te_raw]; te_y = np.array([name2idx[norm(x[1])] for x in te_raw])
    train_classes = sorted(set(tr_y.tolist()))
    log(f"classes={len(class_names)} (in train: {len(train_classes)}) train={len(tr_p)} test={len(te_p)}")

    train_ds = ImgDS(tr_p, tr_y, tf_train)
    sampler = DistributedSampler(train_ds, num_replicas=WORLD, rank=RANK, shuffle=True, drop_last=True, seed=CFG["seed"])
    n_sub = min(CFG["eval_train_subset"] or len(tr_p), len(tr_p))
    return DataBundle(
        class_names, train_classes, tr_d, sampler,
        make_loader(train_ds, CFG["batch_size"], sampler),
        make_loader(shard(ImgDS(tr_p[:n_sub], tr_y[:n_sub], tf_eval)), 32),
        make_loader(shard(ImgDS(te_p, te_y, tf_eval)), 32),
    )
