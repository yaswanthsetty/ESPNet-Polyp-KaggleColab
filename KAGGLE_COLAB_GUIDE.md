# Kaggle / Colab Free-Tier Guide (ESPNet Polyp Segmentation)

## Credits / Attribution

This repository is adapted from the original project:
- https://github.com/Raneem-MT/ESPNet_Polyp_Segmentation

All credit for the method/model goes to the original authors and their MICCAI 2025 paper (see the original repo’s README for citation).

This fork/adaptation only adds practical changes to make training easier on **Kaggle/Colab free tier** (single GPU, mixed precision, gradient accumulation, configurable checkpoint output).

---

## 0) What you need

- A Kaggle or Colab notebook with a **GPU** enabled.
- The **training dataset zip** (download link is in the original README).
- The **pretrained ViT checkpoint** (download link is in the original README).

---

## 1) Dataset folder layout (required)

Training code expects:

```
<TRAIN_ROOT>/
  images/   # RGB images
  masks/    # binary masks (.png)
  edges/    # edge maps (.png) aligned to image filenames
```

Notes:
- `edges/` is required by `dataset.TrainDataset` and used by the edge loss.
- Image/mask/edge filenames must match (same stem).

Testing expects:

```
<TEST_ROOT>/
  images/
  masks/
```

---

## 2) Kaggle: how to upload and place the dataset

Kaggle is most reliable when you **upload the zip as a Kaggle Dataset** and attach it to the notebook.

1) On your local machine: download the training zip.
2) Kaggle → Datasets → New Dataset → upload the zip.
3) Kaggle Notebook → Add Data → attach your dataset.

In the notebook, your data will appear under:

- `/kaggle/input/<your-dataset-name>/...`

Then copy it to the writable working area:

```bash
!mkdir -p /kaggle/working/data
!cp -r /kaggle/input/<your-dataset-name>/* /kaggle/working/data/
```

Set your training root to the folder that contains `images/`, `masks/`, `edges/`.

---

## 3) Colab: how to place the dataset

Options:

- **Upload to Google Drive** and mount Drive, then unzip to `/content`.
- Or use `gdown` if the link is accessible and Colab has internet.

Drive-based example:

```bash
from google.colab import drive
drive.mount('/content/drive')

# Example: copy zip to /content and unzip
!mkdir -p /content/data
!unzip -q "/content/drive/MyDrive/<path-to-your-zip>.zip" -d /content/data
```

---

## 4) Install dependencies

On Kaggle/Colab, torch is usually already installed.

```bash
!pip -q install -r requirements-colab-kaggle.txt
```

---

## 5) Training (stable settings for free tier)

To reduce out-of-memory risk:

- Start with `--batch_size_per_gpu 1`
- Use `--amp` (mixed precision)
- Use `--accum_steps 2` or `4`
- If still OOM, reduce resolution: `--img_size 320` or `352`

Kaggle example:

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

Colab example:

```bash
!python train.py \
  --path "/content/data/<TRAIN_ROOT>" \
  --pretrain "/content/<base_patch16_384.pth>" \
  --output_dir "/content/checkpoints" \
  --batch_size_per_gpu 1 \
  --img_size 352 \
  --num_workers 2 \
  --amp \
  --accum_steps 4
```

Checkpoints will be written to `--output_dir`.

---

## 6) Testing

`test.py` supports passing dataset roots from the command line.

Each dataset root must contain `images/` and `masks/`.

CLI usage:

```bash
!python test.py \
  --ckpt_path "/path/to/model_xx.pth" \
  --result_save_root "/path/to/results" \
  --data_roots "/path/to/dataset1" "/path/to/dataset2" \
  --img_size 352 \
  --threshold 0.5
```
