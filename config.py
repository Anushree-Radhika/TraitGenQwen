TRAIN_JSON = "train.json"
TEST_JSON  = "test.json"
IMAGES_DIR = "CUB_200_2011/images"
OUT_DIR    = "output/"
CKPT_DIR   = f"{OUT_DIR}/checkpoints"
BEST_DIR   = f"{OUT_DIR}/best_model"

MEAN, STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)

CFG = dict(
    img_key="imagePath", desc_key="gt",      # label comes from the class folder name, NOT from gt
    clip="openai/clip-vit-base-patch16",     # "openai/clip-vit-large-patch14" is stronger if memory allows
    llm="Qwen/Qwen3-1.7B",                   # "Qwen/Qwen3-4B" for more capacity (use batch_size=8)
    epochs=12, eval_every=4,
    batch_size=16, grad_accum=1,             # PER GPU -> effective batch = 16 * 2 GPUs * 1 = 32
    lr_clip_lora=1e-4, lr_llm_lora=2e-4, lr_proj=1e-3,
    weight_decay=0.01, warmup_frac=0.05, label_smoothing=0.05, grad_clip=1.0,
    name_weight=1.0,
    desc_weight=0.15,                        # 0 disables description loss
    desc_max_tokens=64,
    aux_weight=0.5,
    lora_r_clip=16, lora_r_llm=32, lora_alpha_mult=2, lora_dropout=0.1,
    grad_ckpt=True, workers=2, eval_train_subset=2000, seed=42, out_dir=OUT_DIR,
)
