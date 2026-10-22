import os
import re
import json
import torch
import random
import logging
import argparse
from pathlib import Path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.dataset import CocoFormatDataset
from PIL import Image
from Model.modelv2 import TraitGen
from Model.VisionEncoderPooled import VisionEncoder
from engine import *

import os
# Force HF to use the standard, stable downloader instead of Xet chunks
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Give the proxy a massive timeout cushion so it doesn't instantly die
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

def get_args_parser():
    parser = argparse.ArgumentParser('XAI Training', add_help=False)
    
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--input_image',default='scratch')
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--encoder_model', default="bioclip",
                        help='Vision encoder: clip | bioclip | bioclip2')
    parser.add_argument('--encoder_op_dim', default=768, type=int)
    parser.add_argument('--decoder_model', default="Qwen/Qwen3-1.7B-Base",
                        help='Decoder LLM: Qwen/Qwen3-1.7B-Base | Qwen/Qwen3-1.7B | openai-community/gpt2-medium')
    parser.add_argument('--streeing_prompt', default="species identification and corresponding textual explanation task.")
    parser.add_argument('--load_path',default="scratch")
    parser.add_argument('--pick_random',default=0,type=int)
    parser.add_argument('--random_dir_path',default="scratch")
    return parser

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    vision_encoder = VisionEncoder(args)
    preprocess = vision_encoder.preprocess
    
    model = TraitGen(args, vision_encoder=vision_encoder).to(device)
    model_state_path = args.load_path
    
    model.eval()
            
            
    if not (model_state_path == "scratch"):
        _ = load_checkpoint(model_state_path,model,None,None)
    
    if(args.pick_random > 0):
        num_files = args.pick_random
        target_dir = Path(args.random_dir_path)
        if not target_dir.is_dir():
            print(f"Error: Directory '{target_dir}' does not exist.")
            sys.exit(1)

        print(f"Scanning '{target_dir}' for images...")

        # Recursively find all matching image files
        all_images = [
            path for path in target_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not all_images:
            print("No image files found.")
            sys.exit(0)

        # Randomly select up to N images
        
        sample_size = min(num_files, len(all_images))
        selected_images = random.sample(all_images, sample_size)

        print(f"Found {len(all_images)} images. Selected {sample_size} at random:\n")

        # Run the python script on each chosen file
        for img in selected_images:
            img_str = str(img)
            infer_image = Image.open(img_str)
            infer_image = preprocess(infer_image)
            infer_image = infer_image.unsqueeze(0).to(device)
            
            prompt_enc = model.decoder.tokenizer(args.streeing_prompt,return_tensors="pt")
            prompt_ids = prompt_enc.input_ids.to(device)
            prompt_mask = prompt_enc.attention_mask.to(device)

            generated_text = model.generate_caption(infer_image,prompt_ids=prompt_ids,prompt_mask=prompt_mask)
            print("="*80)
            print(img_str)
            print(generated_text)
            
            
    
    image_path = args.input_image
    
    
    if not image_path == 'scratch':
        infer_image = Image.open(image_path)
        infer_image = preprocess(infer_image)
        infer_image = infer_image.unsqueeze(0).to(device)
        
        prompt_enc = model.decoder.tokenizer(args.streeing_prompt,return_tensors="pt")
        prompt_ids = prompt_enc.input_ids.to(device)
        prompt_mask = prompt_enc.attention_mask.to(device)

        generated_text = model.generate_caption(infer_image,prompt_ids=prompt_ids,prompt_mask=prompt_mask)
        print(generated_text)
    else:
        print("ERROR - Image path not specified")
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TraitGen Training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)