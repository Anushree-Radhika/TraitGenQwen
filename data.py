import os
import re
import json
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision import transforms as T

from config import CFG, TRAIN_JSON, TEST_JSON, IMAGES_DIR, MEAN, STD
from dist_utils import RANK, WORLD, log


# =============================================================================
# TEXT / JSON HELPERS
# =============================================================================

def norm(s):
    """Normalise a class name: '001.Black_footed_Albatross' -> 'black footed albatross'."""

    s = re.sub(r"^\d+\.", "", str(s).strip())          # drop leading class number
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)

    return s.lower().strip()


def load_json(path):
    """Load records from a .json file or a JSON-lines file. Always returns a list."""

    with open(path) as f:
        text = f.read()

    # Not valid JSON as a whole -> treat as JSON-lines (one record per line)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    # A dict: look for the list of records under a common key
    if isinstance(data, dict):
        for key in ("data", "images", "annotations", "items", "samples", "train", "test"):
            if isinstance(data.get(key), list):
                return data[key]

        # Otherwise assume {image_path: description}
        return [{"imagePath": k, "gt": v} for k, v in data.items()]

    return data


def clean_desc(desc):
    """Remove the leading 'It is a species of "..." as it has' so the label is not leaked."""

    pattern = r'^\s*It is a species of\s*"[^"]*"\s*as it has\s*'

    return re.sub(pattern, "", str(desc or "")).strip()


# =============================================================================
# IMAGE PATH RESOLUTION
# =============================================================================

def index_images():
    """Scan IMAGES_DIR once and build two lookup tables.

    Returns:
        class_dirs : sorted list of class folder names
        idx1       : {filename            -> full path}
        idx2       : {class_dir/filename  -> full path}
    """

    assert os.path.isdir(IMAGES_DIR), f"IMAGES_DIR not found: {IMAGES_DIR}"

    class_dirs = sorted(
        d for d in os.listdir(IMAGES_DIR)
        if os.path.isdir(f"{IMAGES_DIR}/{d}")
    )

    idx1, idx2 = {}, {}

    for class_dir in class_dirs:
        for filename in os.listdir(f"{IMAGES_DIR}/{class_dir}"):
            full_path = f"{IMAGES_DIR}/{class_dir}/{filename}"

            idx1[filename] = full_path
            idx2[f"{class_dir}/{filename}"] = full_path

    return class_dirs, idx1, idx2


def resolve(path, idx1, idx2):
    """Find the real image file for a path taken from the JSON. Returns None if not found."""

    path = str(path).replace("\\", "/")

    # 1) Try the path as-is, then relative to IMAGES_DIR, then to its parent
    candidates = (
        path,
        f"{IMAGES_DIR}/{path}",
        f"{os.path.dirname(IMAGES_DIR)}/{path}",
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    # 2) Fall back to the lookup tables: 'class_dir/filename', then bare filename
    last_two = "/".join(path.split("/")[-2:])

    return idx2.get(last_two) or idx1.get(os.path.basename(path))


def parse(path, idx1, idx2):
    """Read one JSON file and return a list of (image_path, class_folder, description)."""

    records = load_json(path)
    out, missing = [], 0

    log(f"[{os.path.basename(path)}] {len(records)} records | first record: {records[0]}")

    for rec in records:

        # ---- Get the raw image path and description from the record ----
        if isinstance(rec, dict):
            raw_path = rec.get(CFG["img_key"])
            desc     = clean_desc(rec.get(CFG["desc_key"]))
        else:
            raw_path = rec[0] if isinstance(rec, (list, tuple)) else rec
            desc     = ""

        # ---- Resolve to a real file on disk ----
        full_path = resolve(raw_path, idx1, idx2) if raw_path is not None else None

        if full_path is None:
            missing += 1
            continue

        # The label is the class FOLDER name, not the description
        class_folder = os.path.basename(os.path.dirname(full_path))

        out.append((full_path, class_folder, desc))

    log(f"   usable={len(out)} missing_images={missing}")
    assert out, "No images resolved - check IMAGES_DIR and img_key in CFG"

    return out


# =============================================================================
# TRANSFORMS
# =============================================================================

tf_train = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=7),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

tf_eval = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


# =============================================================================
# DATASET / DATALOADERS
# =============================================================================

class ImgDS(Dataset):
    """Returns (image_tensor, class_id, record_index)."""

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")

        return self.transform(image), int(self.labels[i]), i


def collate(batch):
    images  = torch.stack([item[0] for item in batch])
    labels  = torch.tensor([item[1] for item in batch])
    indices = torch.tensor([item[2] for item in batch])

    return images, labels, indices


def make_loader(dataset, batch_size, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=CFG["workers"],
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        drop_last=sampler is not None,      # only drop the last batch when training
    )


def shard(dataset):
    """Give each GPU an exact slice of the data (no padding), so eval metrics are exact."""

    return Subset(dataset, list(range(RANK, len(dataset), WORLD)))


# =============================================================================
# DATA BUNDLE (everything the rest of the code needs)
# =============================================================================

@dataclass
class DataBundle:
    class_names: list                # all class names, index = class id
    train_classes: list              # class ids that appear in the training split
    train_descs: list                # cleaned description for each training record
    sampler: DistributedSampler      # call sampler.set_epoch(ep) each epoch
    train_dl: DataLoader
    train_eval_dl: DataLoader        # subset of train data, no augmentation
    test_dl: DataLoader

    @property
    def ncls(self):
        return len(self.class_names)


def build_data():
    """Parse the JSON files and build every dataloader."""

    # ---------- 1) Index images and build the class list ----------
    class_dirs, idx1, idx2 = index_images()

    class_names = [norm(d) for d in class_dirs]
    name2idx = {name: i for i, name in enumerate(class_names)}

    # ---------- 2) Parse train / test JSON ----------
    train_raw = parse(TRAIN_JSON, idx1, idx2)
    test_raw  = parse(TEST_JSON,  idx1, idx2)

    # Any folder name not seen yet becomes a new class
    for _, label, _ in train_raw + test_raw:
        key = norm(label)

        if key not in name2idx:
            name2idx[key] = len(class_names)
            class_names.append(key)

    # ---------- 3) Split into paths / labels / descriptions ----------
    train_paths = [x[0] for x in train_raw]
    train_y     = np.array([name2idx[norm(x[1])] for x in train_raw])
    train_descs = [x[2] for x in train_raw]

    test_paths  = [x[0] for x in test_raw]
    test_y      = np.array([name2idx[norm(x[1])] for x in test_raw])

    train_classes = sorted(set(train_y.tolist()))

    log(f"classes={len(class_names)} (in train: {len(train_classes)}) "
        f"train={len(train_paths)} test={len(test_paths)}")

    # ---------- 4) Build datasets and loaders ----------
    train_ds = ImgDS(train_paths, train_y, tf_train)

    sampler = DistributedSampler(
        train_ds,
        num_replicas=WORLD,
        rank=RANK,
        shuffle=True,
        drop_last=True,
        seed=CFG["seed"],
    )

    # A fixed slice of the training set, evaluated without augmentation
    n_sub = min(CFG["eval_train_subset"] or len(train_paths), len(train_paths))

    train_eval_ds = ImgDS(train_paths[:n_sub], train_y[:n_sub], tf_eval)
    test_ds       = ImgDS(test_paths, test_y, tf_eval)

    return DataBundle(
        class_names=class_names,
        train_classes=train_classes,
        train_descs=train_descs,
        sampler=sampler,
        train_dl=make_loader(train_ds, CFG["batch_size"], sampler),
        train_eval_dl=make_loader(shard(train_eval_ds), 32),
        test_dl=make_loader(shard(test_ds), 32),
    )
