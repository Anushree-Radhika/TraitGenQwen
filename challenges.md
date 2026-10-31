
## 🩺 Troubleshooting

&nbsp;

**`CUDA out of memory`**
- Lower `batch_size` (try 8), and raise `grad_accum` to keep the effective batch size.
- Keep `grad_ckpt=True`.
- Use the smaller CLIP (`base-patch16`) and Qwen3-1.7B.

&nbsp;

**`IMAGES_DIR not found`**
- The download or extraction cell did not finish. Check that `/kaggle/working/CUB_200_2011/images` exists.

&nbsp;

**`No images resolved - check IMAGES_DIR and img_key`**
- The JSON key for the image path does not match. Open one record in the printed `first record:` line and set `img_key` in `config.py` to the correct key.

&nbsp;

**Lots of `missing_images`**
- Image paths in the JSON do not match the folder layout. The loader tries several fallbacks, but if none work you may need to adjust `resolve(...)` in `data.py`.

&nbsp;

**`ModuleNotFoundError` for one of the project files**
- Make sure you ran `%cd /kaggle/working` before `torchrun`, and that every `%writefile` cell actually ran.

&nbsp;

**`%%writefile` writes an empty file or fails**
- `%%writefile` must be the **first line** of the cell, with nothing above it, not even a comment.

&nbsp;

**Strange import behaviour with `evaluate`**
- `evaluate.py` shares its name with the Hugging Face `evaluate` package. Locally it takes priority, but if you hit odd import errors, rename it to `evaluation.py` and update the import in `train.py`.

&nbsp;

**Only one GPU is used**
- Make sure the notebook accelerator is **GPU T4 x2**, and that you launched with `torchrun --nproc_per_node=2`. Running `python main.py` directly uses a single process.

&nbsp;

**Training hangs at the start**
- Usually a barrier mismatch or a failed weight download on rank 0. Check that internet is enabled and that rank 0's log shows no error.

&nbsp;

**Loss is `nan` early on**
- The `GradScaler` normally recovers on its own. If it persists, lower the learning rates slightly.

&nbsp;

---

&nbsp;
