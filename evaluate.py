import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dist_utils import dev, reduce, MAIN


# =============================================================================
# CHEAP EVAL: name loss + aux-head accuracy (runs every epoch)
# =============================================================================

@torch.no_grad()
def eval_loss(model, dl, codec):
    """Teacher-forced evaluation on the species-name tokens.

    Returns:
        name_loss : mean cross-entropy per name token (summed across GPUs)
        head_acc  : accuracy of the auxiliary classification head
    """

    model.eval()

    # Running totals: [ce_sum, n_tokens, head_correct, n_images]
    totals = [0.0, 0.0, 0.0, 0.0]

    for px, y, _ in dl:

        # ---- Prepare inputs (names only, no description) ----
        px = px.to(dev)
        ans, ans_mask, _, _ = codec.make_ans(y)

        # ---- Forward pass ----
        with torch.autocast("cuda", dtype=torch.float16):
            logits, targets, aux_logits = model(px, ans, ans_mask)

        # ---- Name cross-entropy (summed, divided by token count at the end) ----
        ce_sum = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        n_tokens = (targets != -100).sum()

        # ---- Aux-head accuracy ----
        head_correct = (aux_logits.argmax(1).cpu() == y).sum()

        totals[0] += ce_sum.item()
        totals[1] += n_tokens.item()
        totals[2] += head_correct.item()
        totals[3] += len(y)

    # ---- Combine totals across GPUs ----
    totals = reduce(totals)

    name_loss = totals[0] / totals[1]
    head_acc  = totals[2] / totals[3]

    return name_loss, head_acc


# =============================================================================
# EXPENSIVE EVAL: real generation accuracy (runs every `eval_every` epochs)
# =============================================================================

@torch.no_grad()
def eval_acc(model, dl):
    """Generate a species name for every image (constrained decoding) and score it.

    Returns:
        accuracy : fraction of images where the predicted class id is correct
    """

    model.eval()

    correct = 0
    total   = 0

    for px, y, _ in tqdm(dl, desc="accuracy", leave=False, disable=not MAIN):

        # ---- Generate predictions (list of class ids, -1 if invalid) ----
        with torch.autocast("cuda", dtype=torch.float16):
            preds = model.predict(px.to(dev))

        # ---- Count correct predictions ----
        correct += sum(pred == target for pred, target in zip(preds, y.tolist()))
        total   += len(y)

    # ---- Combine totals across GPUs ----
    correct, total = reduce([correct, total])

    return correct / total
