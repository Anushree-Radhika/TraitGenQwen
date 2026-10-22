import os
import re
import json
import torch
import random
import logging
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.dataset import CocoFormatDataset
from Model.modelv2 import TraitGen
from Model.VisionEncoderPooled import VisionEncoder
from engine import *

def setup_ddp():
    """Initializes the DDP environment."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Destroys the process group upon completion."""
    dist.destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser('XAI Training', add_help=False)

    parser.add_argument('--data_root', default='/home/paul/Paul/DATASETS/cub')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--encoder_model', default="bioclip",
                        help='Vision encoder: clip | bioclip | bioclip2  (or a raw open_clip hub string)')
    parser.add_argument('--encoder_op_dim', default=768, type=int)
    parser.add_argument('--decoder_model', default="Qwen/Qwen3-1.7B-Base",
                        help='Decoder LLM: Qwen/Qwen3-1.7B-Base | Qwen/Qwen3-1.7B | openai-community/gpt2-medium')
    parser.add_argument('--streeing_prompt', default="species identification and corresponding textual explanation task.")
    parser.add_argument('--ann_dir', default='/kaggle/input/custom-ds', help='Path to custom JSON annotations')
    parser.add_argument('--validate_model',default=0,type=int)
    parser.add_argument('--load_path',default="scratch")
    parser.add_argument('--drop_parts',default=2,type=int)
    return parser

def main(args):
    local_rank = setup_ddp()
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # Set up logging ONLY on the primary process (Rank 0)
    logger = None
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "log.txt")
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s | %(message)s')
        logger = logging.getLogger(__name__)

    vision_encoder = VisionEncoder(args)
    preprocess = vision_encoder.preprocess

    img_dir = args.data_root if os.path.basename(args.data_root.rstrip('/\\')) == 'images' else os.path.join(args.data_root, 'images')

    train_dataset = CocoFormatDataset(args, ann_file=f'{args.ann_dir}/cub_train_split1.json', img_prefix=img_dir, preprocess=preprocess)
    test_dataset = CocoFormatDataset(args, ann_file=f'{args.ann_dir}/cub_test_split1.json', img_prefix=img_dir, preprocess=preprocess)

    # Wrap Datasets with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        sampler=test_sampler, 
        num_workers=2,
        pin_memory=True
    )
    
    if(args.validate_model == 1):
        model_state_path = args.load_path
        if not (model_state_path == "scratch"):
        
            model_test = TraitGen(args, vision_encoder=vision_encoder).to(device)
            ckpt_info = load_checkpoint(model_state_path,model_test,None,None)
            model_test = DDP(model_test,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False)
            val_loss,val_acc = validate(args,model_test,test_loader,device)
            if global_rank == 0:
                logger.info(f"Epoch {ckpt_info}: Accuracy={val_acc:.4f} Validation Loss={val_loss:.4f}")
        else:
            print("ERROR. NO PATH MENTIONED FOR LOADING")
        
    else:
        # Initialize model and wrap in DDP
        model = TraitGen(args, vision_encoder=vision_encoder).to(device)
        model_state_path = args.load_path
        
        start_epoch = 0
        
        if not (model_state_path == "scratch"):
            start_epoch = load_checkpoint(model_state_path,model,None,None)
            start_epoch += 1
        
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )    
        
        scaler = torch.cuda.amp.GradScaler()
        
        # ── Track losses for the curve ──
        epoch_list = []
        train_losses = []
        val_losses = []
        
        cnter = 1
        for epoch in range(start_epoch,start_epoch + args.epochs):
            # Set epoch for sampler to ensure proper shuffling across GPUs
            train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch,scaler=scaler)

            # Log and save checkpoints only from rank 0
            if global_rank == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}")

                if epoch == start_epoch + args.epochs - 1:
                    # Save model.module to strip the 'module.' wrapper prefix
                    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
                    save_checkpoint(checkpoint_path, model.module, optimizer,None,epoch)
                elif cnter % 2 == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"best_model_{epoch}.pth")
                    # Save model.module to strip the 'module.' wrapper prefix
                    save_checkpoint(checkpoint_path, model.module, optimizer,None,epoch)
            
            val_loss, val_acc = validate(args, model, test_loader, device)
            if global_rank == 0:
                logger.info(f"Epoch {epoch}: Validation Loss={val_loss:.4f} Validation Accuracy={val_acc:.4f}")
            
            # Record for loss curve
            epoch_list.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
                    
            cnter += 1
        
        # ── Plot loss curve (rank 0 only) ──
        if global_rank == 0 and len(epoch_list) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epoch_list, train_losses, marker='o', linewidth=2, label='Train Loss')
            ax.plot(epoch_list, val_losses, marker='s', linewidth=2, label='Val Loss')
            ax.set_xlabel('Epoch', fontsize=13)
            ax.set_ylabel('Loss', fontsize=13)
            ax.set_title(f'Loss Curve — Encoder: {args.encoder_model}', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            curve_path = os.path.join(args.output_dir, "loss_curve.png")
            fig.savefig(curve_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Loss curve saved to {curve_path}")
            print(f"[INFO] Loss curve saved to {curve_path}")

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TraitGen Training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
