import os, random
import numpy as np, torch
import torch.distributed as dist

RANK  = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
MAIN  = RANK == 0
dev   = torch.device("cuda", LOCAL)

def init_distributed(seed):
    torch.cuda.set_device(LOCAL)
    if WORLD > 1:
        dist.init_process_group("nccl")
    random.seed(seed + RANK); np.random.seed(seed + RANK); torch.manual_seed(seed + RANK)

def cleanup():
    if WORLD > 1:
        dist.destroy_process_group()

def log(*a):
    if MAIN:
        print(*a, flush=True)

def barrier():
    if WORLD > 1:
        dist.barrier()

def reduce(vals):
    t = torch.tensor(vals, dtype=torch.float64, device=dev)
    if WORLD > 1:
        dist.all_reduce(t)
    return t.tolist()
