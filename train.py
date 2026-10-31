import json
import math
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import CFG, CKPT_DIR, BEST_DIR
from dist_utils import dev, MAIN, WORLD, log, barrier, reduce
from evaluate import eval_loss, eval_acc


# =============================================================================
# OPTIMIZER
# =============================================================================

def build_optimizer(model, n_batches):
    """AdamW with three parameter groups (CLIP LoRA / LLM LoRA / projector+head)
    and a cosine schedule with linear warmup."""

    groups = {"clip": [], "llm": [], "proj": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("vision."):
            groups["clip"].append(param)
        elif name.startswith(("proj.", "head.")):
            groups["proj"].append(param)
        else:
            groups["llm"].append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": groups["clip"], "lr": CFG["lr_clip_lora"]},
            {"params": groups["llm"],  "lr": CFG["lr_llm_lora"]},
            {"params": groups["proj"], "lr": CFG["lr_proj"]},
        ],
        weight_decay=CFG["weight_decay"],
    )

    steps_per_epoch = math.ceil(n_batches / CFG["grad_accum"])
    total_steps     = steps_per_epoch * CFG["epochs"]
    warmup_steps    = int(CFG["warmup_frac"] * total_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    all_trainable = [p for group in groups.values() for p in group]

    return optimizer, scheduler, all_trainable


# =============================================================================
# CHECKPOINTS
# =============================================================================

def save_ckpt(model, path, epoch, hist, class_names):
    """Save only the trainable weights (LoRA, projector, head) plus metadata."""

    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    torch.save(
        {
            "epoch": epoch,
            "trainable_state": trainable_state,
            "history": hist,
            "class_names": class_names,
            "cfg": CFG,
        },
        path,
    )


def save_best(model, epoch, hist, class_names):
    """Save the best model in both a single-file and a PEFT-adapter format."""

    save_ckpt(model, f"{BEST_DIR}/best.pt", epoch, hist, class_names)

    model.vision.save_pretrained(f"{BEST_DIR}/clip_lora")
    model.llm.save_pretrained(f"{BEST_DIR}/qwen3_lora")

    torch.save(
        {"proj": model.proj.state_dict(), "head": model.head.state_dict()},
        f"{BEST_DIR}/projector_head.pt",
    )


# =============================================================================
# PLOTTING
# =============================================================================

def plot(hist):
    """Left: losses.  Right: accuracies.  Saved to <out_dir>/loss_curves.png."""

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))

    # ---- Left panel: losses ----
    ax[0].plot(hist["epoch"], hist["train_loss"], "o-",  label="train name loss")
    ax[0].plot(hist["epoch"], hist["test_loss"],  "s-",  label="test name loss")
    ax[0].plot(hist["epoch"], hist["desc_loss"],  "^--", alpha=0.5, label="train desc loss")

    ax[0].set(xlabel="epoch", ylabel="loss (CE / token)", title="Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # ---- Right panel: accuracies ----
    ax[1].plot(hist["epoch"], hist["head_acc"], "^--", alpha=0.6, label="test aux-head acc")

    if hist["acc_epoch"]:
        ax[1].plot(hist["acc_epoch"], hist["train_acc"], "o-", label="train acc")
        ax[1].plot(hist["acc_epoch"], hist["test_acc"],  "s-", label="test acc")

        for epoch, acc in zip(hist["acc_epoch"], hist["test_acc"]):
            ax[1].annotate(
                f"{acc:.3f}", (epoch, acc),
                textcoords="offset points", xytext=(0, 6),
            )

    ax[1].set(
        xlabel="epoch",
        ylabel="accuracy",
        title=f"Accuracy (generation every {CFG['eval_every']} epochs)",
        xlim=(0, CFG["epochs"] + 0.5),
        ylim=(0, 1.05),
    )
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{CFG["out_dir"]}/loss_curves.png', dpi=130)
    plt.close()


# =============================================================================
# ONE TRAINING EPOCH
# =============================================================================

def train_one_epoch(ep, ddp, data, codec, optimizer, scheduler, scaler, all_trainable):
    """Run a single epoch. Returns (name_loss, desc_loss) averaged over all GPUs."""

    accum = CFG["grad_accum"]
    ls    = CFG["label_smoothing"]
    w_name, w_desc, w_aux = CFG["name_weight"], CFG["desc_weight"], CFG["aux_weight"]

    n_batches = len(data.train_dl)

    data.sampler.set_epoch(ep)
    ddp.train()
    optimizer.zero_grad(set_to_none=True)

    # Running totals: [name_ce_sum, name_tokens, desc_ce_sum, desc_tokens]
    running = torch.zeros(4, device=dev, dtype=torch.float64)

    pbar = tqdm(data.train_dl, desc=f"epoch {ep}/{CFG['epochs']}", disable=not MAIN)

    for i, (px, y, record_idx) in enumerate(pbar):

        # Is this the batch where we apply the optimizer step?
        is_step_batch = ((i + 1) % accum == 0) or ((i + 1) == n_batches)

        # ---- Prepare inputs ----
        descs = [codec.desc_tok[j] for j in record_idx.tolist()] if w_desc > 0 else None

        px = px.to(dev, non_blocking=True)
        ans, ans_mask, name_mask, desc_mask = codec.make_ans(y, descs)
        y = y.to(dev)

        # Skip gradient sync on intermediate accumulation micro-batches
        skip_sync = (not is_step_batch) and (WORLD > 1)
        sync_ctx  = ddp.no_sync() if skip_sync else nullcontext()

        # ---- Forward + backward ----
        with sync_ctx:

            with torch.autocast("cuda", dtype=torch.float16):
                logits, targets, aux_logits = ddp(px, ans, ans_mask)

            logits  = logits.float().reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

            per_token = F.cross_entropy(
                logits, targets,
                ignore_index=-100, reduction="none", label_smoothing=ls,
            )

            loss = w_name * per_token[name_mask].mean()
            loss = loss + w_aux * F.cross_entropy(aux_logits.float(), y, label_smoothing=ls)

            if w_desc > 0 and desc_mask.any():
                loss = loss + w_desc * per_token[desc_mask].mean()

            scaler.scale(loss / accum).backward()

        # ---- Track raw (un-smoothed) loss for logging ----
        with torch.no_grad():
            raw = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")

            running[0] += raw[name_mask].sum().double()
            running[1] += name_mask.sum()
            running[2] += raw[desc_mask].sum().double()
            running[3] += desc_mask.sum()

        # ---- Optimizer step ----
        if is_step_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_trainable, CFG["grad_clip"])

            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # If GradScaler skipped the update (inf/NaN grads), skip the LR step too
            if scaler.get_scale() >= prev_scale:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        if MAIN:
            pbar.set_postfix(name_loss=f"{(running[0] / running[1]).item():.4f}")

    # ---- Combine totals across GPUs ----
    totals = reduce(running.tolist())

    name_loss = totals[0] / totals[1]
    desc_loss = totals[2] / max(totals[3], 1)

    return name_loss, desc_loss


# =============================================================================
# FULL TRAINING LOOP
# =============================================================================

def train(model, ddp, data, codec):

    optimizer, scheduler, all_trainable = build_optimizer(model, len(data.train_dl))
    scaler = torch.amp.GradScaler("cuda")

    hist = dict(
        epoch=[], train_loss=[], test_loss=[], desc_loss=[], head_acc=[],
        acc_epoch=[], train_acc=[], test_acc=[],
    )

    best_acc   = -1.0
    start_time = time.time()

    for ep in range(1, CFG["epochs"] + 1):

        # ---------- Train ----------
        train_loss, desc_loss = train_one_epoch(
            ep, ddp, data, codec, optimizer, scheduler, scaler, all_trainable,
        )

        # ---------- Cheap eval (every epoch) ----------
        test_loss, head_acc = eval_loss(model, data.test_dl, codec)

        hist["epoch"].append(ep)
        hist["train_loss"].append(train_loss)
        hist["test_loss"].append(test_loss)
        hist["desc_loss"].append(desc_loss)
        hist["head_acc"].append(head_acc)

        msg = (
            f"[ep {ep}] name_loss train={train_loss:.4f} test={test_loss:.4f} "
            f"gap={test_loss - train_loss:+.4f} | desc_loss={desc_loss:.4f} | head_acc={head_acc:.4f}"
        )

        # ---------- Expensive eval (generation accuracy, every N epochs) ----------
        improved = False

        if ep % CFG["eval_every"] == 0:
            train_acc = eval_acc(model, data.train_eval_dl)
            test_acc  = eval_acc(model, data.test_dl)

            hist["acc_epoch"].append(ep)
            hist["train_acc"].append(train_acc)
            hist["test_acc"].append(test_acc)

            msg += f" | train_acc={train_acc:.4f} test_acc={test_acc:.4f}"

            if test_acc > best_acc:
                best_acc = test_acc
                improved = True
                msg += "  <-- new best"

        log(msg)

        # ---------- Save (rank 0 only) ----------
        if MAIN:
            if improved:
                save_best(model, ep, hist, data.class_names)

            save_ckpt(model, f"{CKPT_DIR}/epoch_{ep:02d}.pt", ep, hist, data.class_names)

            with open(f'{CFG["out_dir"]}/history.json', "w") as f:
                json.dump(hist, f, indent=2)

            plot(hist)

        barrier()

    elapsed_min = (time.time() - start_time) / 60
    log(f"\nDone in {elapsed_min:.1f} min | best test acc = {best_acc:.4f} (saved in {BEST_DIR})")

    return hist
