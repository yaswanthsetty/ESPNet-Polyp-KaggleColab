# ESPNet: Edge-Aware Feature Shrinkage Pyramid for Polyp Segmentation (MICCAI'25)

## Credits

This codebase is based on the original repository:
https://github.com/Raneem-MT/ESPNet_Polyp_Segmentation

If you use this project, please cite the paper (see the Citation section below).

### Architecture
![ESPNet Architecture](figs/architecture.png)

## Training Dataset
- You can download the training data we compiled from here: https://drive.google.com/file/d/16zqdG_3PJaRRkm2wahpE2mp-gyn8rFUF/view?usp=sharing
- You can also download the pretrained transformer here: https://drive.google.com/file/d/1OmE2vEegPPTB1JZpj2SPA6BQnXqiuD1U/view?usp=share_link
- You will need to add their paths to train.sh in scripts.
- When creating edge maps, we recommend dilating them; it produces better results.

## Evaluation
- You need to add the test datasets' paths in test.py, and the model checkpoint and results paths in test.sh (or modify test.py).
- You can download the checkpoint from here: https://drive.google.com/file/d/161dyhkMQtXeF4EbykIGGMbXLBQ16Qbfp/view?usp=drive_link

## Prerequisites

- Creating a virtual environment in terminal: `conda create -n ESPNet python=3.8`.
- Versions: torch==2.4.1 scipy==1.10.1 torchvision==0.7.0 timm==0.5.4

## Running on Kaggle / Colab (free tier)

See also: `KAGGLE_COLAB_GUIDE.md` for a step-by-step, copy/paste guide.

This repo was originally set up for Slurm + multi-GPU (DDP). It now supports **single-GPU** training via `train.py`.

### 1) Dataset layout (required)

Training expects the following directory structure:

```
<TRAIN_ROOT>/
  images/   # RGB images
  masks/    # binary masks (.png)
  edges/    # edge maps (.png) matching image names
```

Testing expects:

```
<TEST_ROOT>/
  images/
  masks/
```

### 2) Getting the dataset into Kaggle

Recommended on Kaggle (more reliable than downloading from Google Drive inside the notebook):

1. Download the training zip from the link above on your local machine.
2. Create a Kaggle Dataset from that zip and attach it to your notebook.
3. In the notebook, the files appear under `/kaggle/input/<your-dataset-name>/`.

Then copy/unzip it into the writable working directory:

```bash
!mkdir -p /kaggle/working/data
!cp -r /kaggle/input/<your-dataset-name>/* /kaggle/working/data/
```

Set:

- `TRAIN_ROOT=/kaggle/working/data/<the-folder-containing-images-masks-edges>`
- `PRETRAIN=/kaggle/input/<pretrained-vit-dataset>/base_patch16_384.pth`

### 3) Install dependencies (Kaggle/Colab)

In a fresh notebook, install missing python packages (Torch is usually preinstalled):

```bash
!pip -q install timm==0.5.4 opencv-python-headless imageio tqdm scipy
```

### 4) Training command (safe defaults for free tier)

To avoid OOM / RAM crashes on free-tier GPUs, start with:

- `--batch_size_per_gpu 1`
- `--amp` (mixed precision)
- `--accum_steps 2` or `4` (simulates larger batch without extra GPU memory)
- optionally reduce resolution with `--img_size 320` or `352`

Example (Kaggle):

```bash
!python train.py \
  --path "/kaggle/working/data/<TRAIN_ROOT>" \
  --pretrain "/kaggle/input/<pretrained-vit-dataset>/base_patch16_384.pth" \
  --output_dir "/kaggle/working/checkpoints" \
  --batch_size_per_gpu 1 \
  --img_size 352 \
  --num_workers 2 \
  --amp \
  --accum_steps 4
```

Checkpoints will be saved under `/kaggle/working/checkpoints/`.

### 5) Testing

`test.py` currently has test dataset roots hardcoded inside the file. For Kaggle/Colab, edit them or adapt `test.py` to point to your attached test datasets.
  

## Results

### Seen Datasets
![Seen Dataset Results](figs/results_seen.png)

### Unseen Datasets
![Unseen Dataset Results](figs/results_unseen.png)


## 📄 Citation

If you use this code or find our work helpful, please cite our paper:

> Raneem Toman, Venkataraman Subramanian, and Sharib Ali.  
> "**ESPNet: Edge-Aware Feature Shrinkage Pyramid for Polyp Segmentation**".  
> *Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2025.

### BibTeX
```bibtex
@inproceedings{toman2025espnet,
  author    = {Raneem Toman and Venkataraman Subramanian and Sharib Ali},
  title     = {ESPNet: Edge-Aware Feature Shrinkage Pyramid for Polyp Segmentation},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2025},
}
