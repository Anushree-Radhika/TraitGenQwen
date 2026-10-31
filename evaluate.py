%%writefile /kaggle/working/evaluate.py
import torch, torch.nn.functional as F
from tqdm.auto import tqdm
from dist_utils import dev, reduce, MAIN

@torch.no_grad()
def eval_loss(model, dl, codec):
    model.eval(); s = [0.0, 0.0, 0.0, 0.0]               # ce_sum, tokens, head_correct, n
    for px, y, _ in dl:
        px = px.to(dev); a, m, nm, dm = codec.make_ans(y)
        with torch.autocast("cuda", dtype=torch.float16): lg, tgt, aux = model(px, a, m)
        s[0] += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum").item()
        s[1] += (tgt != -100).sum().item(); s[2] += (aux.argmax(1).cpu() == y).sum().item(); s[3] += len(y)
    s = reduce(s); return s[0] / s[1], s[2] / s[3]

@torch.no_grad()
def eval_acc(model, dl):
    model.eval(); c = n = 0
    for px, y, _ in tqdm(dl, desc="accuracy", leave=False, disable=not MAIN):
        with torch.autocast("cuda", dtype=torch.float16): pr = model.predict(px.to(dev))
        c += sum(p == t for p, t in zip(pr, y.tolist())); n += len(y)
    c, n = reduce([c, n]); return c / n
