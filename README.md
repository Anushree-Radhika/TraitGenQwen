## Installation

Create and activate the Conda environment:

```bash
conda create -n gen python=3.13 -y
conda activate gen
```

Check that the server GPUs are visible:

```bash
nvidia-smi
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify that PyTorch can see the GPUs:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

> If the GPU count prints `0`, install the PyTorch build that matches your server's CUDA version
> from https://pytorch.org/get-started/locally/, then re-run the command above.

## Dataset

Download and extract CUB-200-2011:

```bash
mkdir -p data && cd data
wget "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" -O CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
cd ..
```

Place `train.json` and `test.json` somewhere on the server, for example in `data/`.

## Configuration

Open `config.py` and set the four paths for your server:

```python
TRAIN_JSON = "/path/to/data/train.json"
TEST_JSON  = "/path/to/data/test.json"
IMAGES_DIR = "/path/to/data/CUB_200_2011/images"
OUT_DIR    = "/path/to/output"
```

Hyperparameters (epochs, learning rates, batch size, model choice) are in the `CFG` dictionary in the
same file. Note that `batch_size` is **per GPU**.

## Training

Start training on 2 GPUs:

```bash
torchrun --nproc_per_node=2 main.py
```

Use specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py
```

Run on a single GPU:

```bash
torchrun --nproc_per_node=1 main.py
```

> With one GPU the effective batch size halves (16 instead of 32). Set `grad_accum=2` in `config.py`
> to keep it at 32.

Run in the background so training survives a disconnect:

```bash
nohup torchrun --nproc_per_node=2 main.py > train.log 2>&1 &
tail -f train.log
```

(Or start it inside `tmux` or `screen`.)

The first run downloads the CLIP and Qwen3 weights from Hugging Face, so the server needs internet access
at that point.

## Outputs

Logs, checkpoints and plots are saved under `OUT_DIR`:

| Path | Contents |
|------|----------|
| `checkpoints/epoch_XX.pt` | Weights for every epoch |
| `best_model/` | Best model by test accuracy |
| `history.json` | Metrics per epoch |
| `loss_curves.png` | Loss and accuracy plots |
