# 🐦 CUB-200 Bird Species VLM

### CLIP vision encoder (LoRA) + Qwen3 decoder (LoRA), trained with DDP on 2× Kaggle T4

&nbsp;

A small vision-language model that looks at a bird photo and **writes out the species name** as text.
The vision side is a CLIP ViT, the language side is Qwen3, and both are fine-tuned with LoRA so the
whole thing fits on two 16 GB T4 GPUs.

At inference time, decoding is **constrained**: the model can only produce a valid class name from the
training set. It can never invent a species that does not exist.

&nbsp;

---

&nbsp;

## 📑 Table of Contents

1. [What this project does](#-what-this-project-does)
2. [How the model works](#-how-the-model-works)
3. [Project structure](#-project-structure)
4. [What each file does](#-what-each-file-does)
5. [Requirements](#-requirements)
6. [Dataset format](#-dataset-format)
7. [Configuration reference](#-configuration-reference)
8. [Training details](#-training-details)
9. [Outputs](#-outputs)
10. [Loading the trained model](#-loading-the-trained-model)
11. [Common tweaks](#-common-tweaks)
&nbsp;

---

&nbsp;

## 🎯 What this project does

**Input:** a photo of a bird.

**Output:** the species name, for example `laysan albatross`.

**Training data:** the CUB-200-2011 images, plus a `train.json` / `test.json` pair that gives each image
path and a text description of the bird.

The label always comes from the **class folder name** (for example `001.Black_footed_Albatross`).
The description field (`gt`) is used only as a light auxiliary training signal.

&nbsp;

---

&nbsp;

## 🧠 How the model works

&nbsp;

### The big picture

```
                 ┌─────────────────────────────┐
  bird photo ──► │  CLIP ViT-B/16  (+ LoRA)    │
  (224 × 224)    └──────────────┬──────────────┘
                                │  197 tokens (1 CLS + 14×14 patches)
                                ▼
                    2×2 average pool on patches
                                │  50 tokens (1 CLS + 7×7 patches)
                                ▼
                 ┌─────────────────────────────┐
                 │  Projector                   │
                 │  LayerNorm → Linear → GELU   │
                 │            → Linear          │
                 └──────────────┬──────────────┘
                                │  50 "visual tokens" in the LLM's space
                                ▼
   text prompt embeddings ──► [ visual tokens | prompt ] ──► Qwen3 (+ LoRA)
   "Question: Which bird                                          │
    species is in this                                            ▼
    image?\nAnswer:"                          " laysan albatross\nDescription: ..."
```

&nbsp;

### Step by step

| Step | What happens |
|------|--------------|
| **1. Encode** | CLIP turns the image into a sequence of patch features. |
| **2. Compress** | The 14×14 patch grid is average-pooled 2×2 to 7×7, so the LLM sees 50 visual tokens instead of 197. This saves a lot of memory. |
| **3. Project** | A small MLP maps CLIP's feature size into Qwen3's embedding size. |
| **4. Prompt** | The visual tokens are placed **before** a fixed text prompt. |
| **5. Generate** | Qwen3 writes the species name, followed by `\nDescription:` and (during training only) a description. |

&nbsp;

### The three losses

Training combines three signals:

| Loss | Weight (`CFG` key) | Applied to | Purpose |
|------|--------------------|------------|---------|
| **Name loss** | `name_weight = 1.0` | The species-name tokens | The **main objective**. |
| **Description loss** | `desc_weight = 0.15` | The description tokens | A light auxiliary signal that pushes the vision features to capture visible traits. Set to `0` to disable. |
| **Aux classification loss** | `aux_weight = 0.5` | A linear head on CLIP's pooled feature | Gives the vision encoder a direct, easy gradient. |

```
total_loss = 1.0 × name_CE  +  0.5 × aux_head_CE  +  0.15 × description_CE
```

All three use `label_smoothing = 0.05`.

&nbsp;

### Constrained decoding (the trie)

Every training class name is tokenized, and all of those token sequences are stored in a **prefix tree (trie)**.
While generating, `prefix_allowed_tokens_fn` looks at the tokens produced so far and only allows
tokens that continue a valid class name.

Result: the output is **always** one of the training classes. If generation ever wanders off the trie,
the prediction is counted as `-1` (wrong).

&nbsp;

---

&nbsp;

## 📁 Project structure

```
/working-directory/
│
├── config.py          ← paths, hyperparameters, constants
├── dist_utils.py      ← DDP setup, logging, barrier, reduce
├── data.py            ← JSON parsing, class names, datasets, dataloaders
├── tokens.py          ← tokenizer, prompt, trie, target building
├── model.py           ← the VLM itself
├── evaluate.py        ← validation loss + generation accuracy
├── train.py           ← optimizer, training loop, checkpoints, plots
├── main.py            ← entry point that wires everything together
│_
/data/
├── CUB_200_2011/      ← dataset (downloaded in setup cell)
│   └── images/
│-test.json
|-train.json
-----
/output/
├── checkpoints/       ← one file per epoch          (created at run time)
├── best_model/        ← best-by-test-accuracy model (created at run time)
├── history.json       ← metrics per epoch           (created at run time)
└── loss_curves.png    ← loss and accuracy plots     (created at run time)
```

&nbsp;

---

&nbsp;

## 🧩 What each file does

&nbsp;

### `config.py` , *settings only*

Holds every path and every hyperparameter in one place (`CFG`). **No logic lives here.**
If you want to change a setting, this is the only file you should need to open.

&nbsp;

### `dist_utils.py` , *multi-GPU plumbing*

| Name | Purpose |
|------|---------|
| `RANK`, `WORLD`, `LOCAL` | Process id, number of processes, GPU index (read from `torchrun` env vars). |
| `MAIN` | `True` only on rank 0. Used to guard printing and saving. |
| `dev` | The CUDA device for this process. |
| `init_distributed(seed)` | Sets the GPU, starts NCCL, seeds all RNGs (seed + rank). |
| `log(...)` | `print` that only runs on rank 0. |
| `barrier()` | Wait for all processes. |
| `reduce(vals)` | Sum a list of numbers across GPUs. |
| `cleanup()` | Shuts down the process group. |

&nbsp;

### `data.py` , *everything about the dataset*

| Piece | Purpose |
|-------|---------|
| `norm`, `load_json`, `clean_desc` | Small helpers to normalise class names, read JSON or JSONL, and strip the boilerplate prefix from descriptions. |
| `index_images`, `resolve`, `parse` | Find the real image file for each JSON record and attach label and description. |
| `tf_train`, `tf_eval` | Augmented transforms for training, plain resize and center-crop for evaluation. |
| `ImgDS` | Dataset that returns `(image, class_id, record_index)`. |
| `shard` | Splits eval data across GPUs **without padding**, so metrics are exact. |
| `DataBundle` | A small dataclass that carries everything downstream code needs. |
| `build_data()` | The one function `main.py` calls. |

&nbsp;

### `tokens.py` , *text side*

The `TextCodec` class owns:

- the Qwen3 tokenizer and its `eos` / `pad` ids,
- the tokenized class names (plus the `\nDescription:` separator),
- the tokenized training descriptions,
- the fixed question prompt,
- the **trie** and the lookup table used for constrained decoding,
- `make_ans(...)`, which builds padded target tensors and the **name mask** / **description mask**.

&nbsp;

### `model.py` , *the network*

The `VLM` class:

- loads CLIP vision and Qwen3 in fp16 and wraps each with LoRA,
- enables gradient checkpointing,
- builds the projector MLP and the auxiliary classification head,
- exposes `forward(...)` for training and `predict(...)` for constrained generation.

&nbsp;

### `evaluate.py` , *measuring quality*

| Function | What it reports | Cost |
|----------|-----------------|------|
| `eval_loss` | Name cross-entropy and aux-head accuracy | Cheap, runs **every epoch** |
| `eval_acc` | True generation accuracy using the trie | Slower, runs **every `eval_every` epochs** |

Both are sharded across GPUs and combined with `reduce`.

&nbsp;

### `train.py` , *the training loop*

Contains the optimizer builder (three parameter groups with separate learning rates), the cosine schedule
with warmup, checkpoint saving, the loss-curve plot, and the `train(...)` function itself.

&nbsp;

### `main.py` , *the conductor*

Short on purpose. In order, it:

1. initialises distributed training,
2. builds the data,
3. builds the codec and the model (rank 0 first, so weights download only once),
4. casts trainable weights to fp32,
5. wraps the model in DDP,
6. calls `train(...)`,
7. cleans up.

&nbsp;

---

&nbsp;

## 📦 Requirements

| Requirement | Version / note |
|-------------|----------------|
| Python | 3.10+ |
| PyTorch | with CUDA and NCCL |
| `transformers` | **≥ 4.51.0** (needed for Qwen3) |
| `peft` | recent release |
| `torchvision`, `Pillow`, `numpy`, `matplotlib`, `tqdm` | pre-installed on Kaggle |
| `torchao` | `> 0.16.0` (used in your setup cell) |
| Hardware | **2× NVIDIA T4 (16 GB each)**, as on Kaggle |

Install:

```bash
pip install -q -U "transformers>=4.51.0" peft
pip install --upgrade "torchao>0.16.0"
```

&nbsp;

---

&nbsp;

## 🗂 Dataset format

&nbsp;

### Images

The standard CUB-200-2011 layout:

```
CUB_200_2011/images/
├── 001.Black_footed_Albatross/
│   ├── Black_Footed_Albatross_0001_796111.jpg
│   └── ...
├── 002.Laysan_Albatross/
└── ...
```

&nbsp;

### `train.json` and `test.json`

Each file is a list of records (JSON or JSON-lines). Every record needs:

| Key | Meaning |
|-----|---------|
| `imagePath` | Path to the image. It may be absolute, relative to `IMAGES_DIR`, or just `Class_Folder/filename.jpg`. A bare filename also works. |
| `gt` | A text description of the bird. Used only for the light description loss. |

Example:

```json
{
  "imagePath": "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg",
  "gt": "It is a species of \"Black footed Albatross\" as it has a long hooked bill and dark brown plumage."
}
```

The description cleaner removes the leading `It is a species of "..." as it has` part, so the model is
not trained to copy the label from the description.

> 💡 The loader tries several path-resolution strategies. Any record whose image cannot be found is
> **skipped**, and the number of skipped records is printed at startup (`missing_images=...`).

&nbsp;

---

&nbsp;
### Precision

- Frozen base weights are stored in **fp16**.
- Trainable weights (LoRA, projector, head) are cast to **fp32** so `GradScaler` works correctly.
- Forward passes run under `torch.autocast(float16)`.

&nbsp;

### Optimizer and schedule

- **AdamW** with three parameter groups (CLIP LoRA, LLM LoRA, projector and head).
- **Cosine** decay with linear warmup.
- If `GradScaler` skips a step because of inf/NaN gradients, the scheduler step is skipped too, so the LR schedule stays aligned with real updates.

&nbsp;

### Multi-GPU behaviour

- `DistributedSampler` splits training data across the two GPUs (`drop_last=True`).
- Evaluation is sharded **without padding**, so every test image is counted exactly once.
- Metrics are summed across GPUs with `reduce(...)` before being logged.
- With `grad_accum > 1`, `ddp.no_sync()` avoids useless gradient syncs on intermediate micro-batches.
- Rank 0 loads model weights first, then other ranks read from the cache. This prevents two processes downloading the same files at once.

&nbsp;

### What gets evaluated, and when

| Metric | Frequency | Meaning |
|--------|-----------|---------|
| Train name loss | every epoch | Running average over the epoch |
| Test name loss | every epoch | Cross-entropy on the species-name tokens |
| Aux-head accuracy | every epoch | Accuracy of the linear head on the pooled CLIP feature |
| Description loss | every epoch | Training-time only |
| **Generation accuracy** (train subset and test) | every `eval_every` epochs | The true end-to-end metric, using constrained decoding |

&nbsp;

---

&nbsp;

## 📤 Outputs

| Path | Contents |
|------|----------|
| `checkpoints/epoch_XX.pt` | Trainable weights, history, class names and config for every epoch. |
| `best_model/best.pt` | Same format, saved whenever **test generation accuracy** improves. |
| `best_model/clip_lora/` | CLIP LoRA adapter (PEFT format). |
| `best_model/qwen3_lora/` | Qwen3 LoRA adapter (PEFT format). |
| `best_model/projector_head.pt` | Projector and aux-head `state_dict`s. |
| `history.json` | All metrics per epoch. |
| `loss_curves.png` | Left: losses. Right: accuracies. Updated each epoch. |

> ⚠️ The "best" model is chosen using **test** accuracy, since there is no separate validation split.
> Treat the reported best score as slightly optimistic.

&nbsp;

---

&nbsp;

## 🔄 Loading the trained model

A minimal outline for inference:

```python
import torch
from config import CFG
from data import build_data
from tokens import TextCodec
from model import VLM
from dist_utils import dev

ckpt = torch.load("/kaggle/working/best_model/best.pt", map_location="cpu")
class_names = ckpt["class_names"]

# Rebuild the codec and model with the same class list
# (train_classes / train_descs only matter for the trie and training targets)
codec = TextCodec(class_names, train_classes=list(range(len(class_names))), train_descs=[])
model = VLM(len(class_names), codec).to(dev)

# Load only the trainable weights that were saved
missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
model.eval()

# pred_ids = model.predict(pixel_batch)   # list of class ids, -1 if invalid
# names    = [class_names[i] for i in pred_ids if i >= 0]
```

> 💡 Pass the **same** `train_classes` you trained with if you want the trie to match training exactly.
> The `class_names` saved in the checkpoint give you the full class list.

&nbsp;

---

&nbsp;

## 🔧 Common tweaks

| I want to... | Change this |
|--------------|-------------|
| Use a bigger vision encoder | `clip` in `config.py` |
| Use a bigger LLM | `llm` in `config.py` (and lower `batch_size` to 8) |
| Train longer | `epochs` |
| See accuracy more often | Lower `eval_every` (generation is slow) |
| Turn off description loss | `desc_weight = 0` |
| Reduce GPU memory | Lower `batch_size`, raise `grad_accum`, keep `grad_ckpt=True` |
| Change the prompt wording | `TextCodec.__init__` in `tokens.py` |
| Change how visual tokens are compressed | `VLM.encode_image` in `model.py` |
| Try a different loss or LR schedule | `train.py` only |
| Train on a different dataset | `data.py` and the paths in `config.py` |

&nbsp;

---

&nbsp;
---

&nbsp;

<sub>Built for Kaggle 2× T4 · PyTorch DDP · PEFT LoRA · Hugging Face Transformers</sub>##
