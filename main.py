import os

# Must be set before the tokenizer library is imported/used
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from config import CFG, CKPT_DIR, BEST_DIR
from dist_utils import (
    RANK, WORLD, LOCAL, MAIN, dev,
    init_distributed, cleanup, log, barrier,
)
from data import build_data
from tokens import TextCodec
from model import VLM
from train import train


def count_params(model):
    """Return (trainable, total) parameter counts."""

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    return trainable, total


def main():

    # ---------- 1) Distributed setup ----------
    init_distributed(CFG["seed"])
    log(f"GPUs visible: {torch.cuda.device_count()} | world_size={WORLD}")

    # Only rank 0 creates the output folders
    if MAIN:
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(BEST_DIR, exist_ok=True)

    # ---------- 2) Data ----------
    data = build_data()

    # ---------- 3) Tokenizer + model ----------
    # Rank 0 downloads the weights first; the other ranks wait, then read the cache.
    if RANK != 0:
        barrier()

    codec = TextCodec(data.class_names, data.train_classes, data.train_descs)
    model = VLM(data.ncls, codec).to(dev)

    # Trainable weights (LoRA / projector / head) must be fp32 for GradScaler
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Rank 0 is done loading -> release the waiting ranks
    if RANK == 0:
        barrier()

    n_trainable, n_total = count_params(model)
    log(
        f"trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M "
        f"({100 * n_trainable / n_total:.2f}%)"
    )

    # ---------- 4) Wrap in DDP (only when using more than one GPU) ----------
    if WORLD > 1:
        ddp = DDP(model, device_ids=[LOCAL], broadcast_buffers=False)
    else:
        ddp = model

    # ---------- 5) Train ----------
    train(model, ddp, data, codec)

    # ---------- 6) Clean up ----------
    cleanup()


if __name__ == "__main__":
    main()
