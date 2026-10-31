%%writefile /kaggle/working/train.py
import json, math, time, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import nullcontext
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from config import CFG, CKPT_DIR, BEST_DIR
from dist_utils import dev, MAIN, WORLD, log, barrier, reduce
from evaluate import eval_loss, eval_acc

def build_optimizer(model, n_batches):
    groups = {"clip": [], "llm": [], "proj": []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups["clip" if n.startswith("vision.") else "proj" if n.startswith(("proj.", "head.")) else "llm"].append(p)
    opt = torch.optim.AdamW([{"params": groups["clip"], "lr": CFG["lr_clip_lora"]},
                             {"params": groups["llm"],  "lr": CFG["lr_llm_lora"]},
                             {"params": groups["proj"], "lr": CFG["lr_proj"]}], weight_decay=CFG["weight_decay"])
    total = math.ceil(n_batches / CFG["grad_accum"]) * CFG["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(CFG["warmup_frac"] * total), total)
    return opt, sched, [p for g in groups.values() for p in g]

def save_ckpt(model, path, epoch, hist, class_names):
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"epoch": epoch, "trainable_state": state, "history": hist, "class_names": class_names, "cfg": CFG}, path)

def save_best(model, epoch, hist, class_names):
    save_ckpt(model, f"{BEST_DIR}/best.pt", epoch, hist, class_names)
    model.vision.save_pretrained(f"{BEST_DIR}/clip_lora"); model.llm.save_pretrained(f"{BEST_DIR}/qwen3_lora")
    torch.save({"proj": model.proj.state_dict(), "head": model.head.state_dict()}, f"{BEST_DIR}/projector_head.pt")

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

def train(model, ddp, data, codec):
    opt, sched, all_trainable = build_optimizer(model, len(data.train_dl))
    scaler = torch.amp.GradScaler("cuda")
    hist = dict(epoch=[], train_loss=[], test_loss=[], desc_loss=[], head_acc=[], acc_epoch=[], train_acc=[], test_acc=[])
    best_acc, t0, A = -1.0, time.time(), CFG["grad_accum"]
    LS, DW, NW, AW = CFG["label_smoothing"], CFG["desc_weight"], CFG["name_weight"], CFG["aux_weight"]

    for ep in range(1, CFG["epochs"] + 1):
        data.sampler.set_epoch(ep); ddp.train()
        run = torch.zeros(4, device=dev, dtype=torch.float64)       # name_ce_sum, name_tokens, desc_ce_sum, desc_tokens
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(data.train_dl, desc=f"epoch {ep}/{CFG['epochs']}", disable=not MAIN)
        for i, (px, y, ridx) in enumerate(pbar):
            step_now = (i + 1) % A == 0 or (i + 1) == len(data.train_dl)
            descs = [codec.desc_tok[j] for j in ridx.tolist()] if DW > 0 else None
            px = px.to(dev, non_blocking=True); a, m, nm, dm = codec.make_ans(y, descs); y = y.to(dev)
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
        te_loss, head_acc = eval_loss(model, data.test_dl, codec)
        hist["epoch"].append(ep); hist["train_loss"].append(tr_loss); hist["test_loss"].append(te_loss)
        hist["desc_loss"].append(d_loss); hist["head_acc"].append(head_acc)
        msg = f"[ep {ep}] name_loss train={tr_loss:.4f} test={te_loss:.4f} gap={te_loss-tr_loss:+.4f} | desc_loss={d_loss:.4f} | head_acc={head_acc:.4f}"
        improved = False
        if ep % CFG["eval_every"] == 0:
            tr_acc, te_acc = eval_acc(model, data.train_eval_dl), eval_acc(model, data.test_dl)
            hist["acc_epoch"].append(ep); hist["train_acc"].append(tr_acc); hist["test_acc"].append(te_acc)
            msg += f" | train_acc={tr_acc:.4f} test_acc={te_acc:.4f}"
            if te_acc > best_acc: best_acc, improved = te_acc, True; msg += "  <-- new best"
        log(msg)
        if MAIN:
            if improved: save_best(model, ep, hist, data.class_names)
            save_ckpt(model, f"{CKPT_DIR}/epoch_{ep:02d}.pt", ep, hist, data.class_names)
            json.dump(hist, open(f'{CFG["out_dir"]}/history.json', "w"), indent=2); plot(hist)
        barrier()
    log(f"\nDone in {(time.time()-t0)/60:.1f} min | best test acc = {best_acc:.4f} (saved in {BEST_DIR})")
    return hist
