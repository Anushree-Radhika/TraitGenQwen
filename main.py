%%writefile /kaggle/working/main.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from config import CFG, CKPT_DIR, BEST_DIR
from dist_utils import RANK, WORLD, LOCAL, MAIN, dev, init_distributed, cleanup, log, barrier
from data import build_data
from tokens import TextCodec
from model import VLM
from train import train

def main():
    init_distributed(CFG["seed"])
    log(f"GPUs visible: {torch.cuda.device_count()} | world_size={WORLD}")
    if MAIN:
        os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(BEST_DIR, exist_ok=True)

    data = build_data()

    if RANK != 0: barrier()            # rank 0 downloads weights first, the others read the cache
    codec = TextCodec(data.class_names, data.train_classes, data.train_descs)
    model = VLM(data.ncls, codec).to(dev)
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()      # fp32 trainable weights (LoRA/proj/head) for GradScaler
    if RANK == 0: barrier()

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"trainable params: {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M ({100*n_tr/n_all:.2f}%)")

    ddp = DDP(model, device_ids=[LOCAL], broadcast_buffers=False) if WORLD > 1 else model
    train(model, ddp, data, codec)
    cleanup()

if __name__ == "__main__":
    main()
