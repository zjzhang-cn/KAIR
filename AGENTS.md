# AGENTS.md — KAIR

## Project overview

KAIR is a research codebase for image/video restoration (denoising, super-resolution, deblurring, JPEG deblocking, frame interpolation). It bundles training and testing scripts for many models: DnCNN, FFDNet, SRMD, DPSR, MSRResNet, RRDB, SwinIR, VRT, RVRT, USRNet, DRUNet, IMDN, BSRGAN. Built on PyTorch with a config-driven, BasicSR-style architecture.

## Developer commands

```bash
# Install
pip install -r requirement.txt

# Download all pretrained models into model_zoo/
python main_download_pretrained_models.py --models "all" --model_dir "model_zoo"

# Download a specific model family
python main_download_pretrained_models.py --models "DnCNN" --model_dir "model_zoo"
```

### Training

```bash
# Single GPU (PSNR-oriented, DataParallel)
python main_train_psnr.py --opt options/train_msrresnet_psnr.json

# GAN training (single GPU)
python main_train_gan.py --opt options/train_msrresnet_gan.json

# Distributed training (multi-GPU)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt options/train_msrresnet_psnr.json --dist True
```

### Testing

Test scripts are per-model. Examples:

```bash
# DnCNN denoising
python main_test_dncnn.py --model_name dncnn_25 --testset_name set12 --noise_level_img 25

# SwinIR (auto-downloads model & dataset if missing)
python main_test_swinir.py --task classical_sr --scale 4 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --folder_lq testsets/set5/LR_bicubic/X4 --folder_gt testsets/set5/HR

# VRT video SR (auto-downloads model & dataset)
python main_test_vrt.py --task 001_VRT_videosr_bi_REDS_6frames --folder_lq testsets/REDS4/sharp_bicubic --folder_gt testsets/REDS4/GT --tile 40 128 128 --tile_overlap 2 20 20

# RVRT video SR
python main_test_rvrt.py --task 001_RVRT_videosr_bi_REDS_30frames --folder_lq testsets/REDS4/sharp_bicubic --folder_gt testsets/REDS4/GT --tile 100 128 128 --tile_overlap 2 20 20
```

### Benchmarking a model
```bash
python main_challenge_sr.py   # FLOPs, #Params, runtime, activations
```

## Architecture

### Entry-point scripts (root level)

- `main_train_psnr.py` — PSNR-driven (pixel loss, `model: plain`)
- `main_train_gan.py` — GAN-based (pixel + perceptual + adversarial, `model: gan`)
- `main_train_vrt.py` — video restoration transformer (uses `model_vrt.py`, `model: vrt`)
- `main_train_usrnet.py`, `main_train_drunet.py` — USRNet/DRUNet training
- `main_test_*.py` — one per model, standalone with hardcoded architecture params
- `main_download_pretrained_models.py` — fetches weights from GitHub releases

### Key directories

| Dir | Purpose |
|-----|---------|
| `models/` | Network definitions (`network_*.py`), loss functions (`loss.py`, `loss_ssim.py`), model wrappers (`model_plain.py`, `model_gan.py`, `model_vrt.py`), factory (`select_model.py`, `select_network.py`) |
| `data/` | Dataset classes (`dataset_*.py`), factory (`select_dataset.py`), video meta-info files (`meta_info/`) |
| `utils/` | Image I/O (`utils_image.py`), option parsing (`utils_option.py`), model utils (`utils_model.py`, `utils_bnorm.py`), distributed init (`utils_dist.py`) |
| `options/` | JSON config files per model+task (subdirs: `swinir/`, `vrt/`, `rvrt/`) |
| `model_zoo/` | Downloaded pretrained weights (subdirs: `swinir/`, `vrt/`, `rvrt/`) |
| `trainsets/`, `testsets/` | Training and testing datasets (not versioned) |
| `docs/` | Detailed READMEs for SwinIR, VRT, RVRT training |
| `kernels/` | Custom CUDA kernels |
| `matlab/` | MATLAB utilities for result visualization |
| `scripts/` | Data preparation utilities (e.g., `create_lmdb.py`) |

### Config system (`utils_option.py`)

- JSON configs allow `//` comments — the parser strips them before `json.loads`
- After parsing, the config is converted to a `NoneDict` — accessing missing keys returns `None` (no `KeyError`)
- Top-level `scale` and `n_channels` are broadcast to dataset and network configs
- `CUDA_VISIBLE_DEVICES` is set from `gpu_ids` during option parsing
- Checkpoints auto-resume: the training script scans `{root}/{task}/models/` for the highest-iteration `.pth` file
- Distributed training defaults: `find_unused_parameters: False`, `use_static_graph: False`

### Model/dataset/network factories

- **Model type** (`models/select_model.py`): `"plain"` (1 input), `"plain2"` (2 inputs), `"plain4"` (4 inputs, USRNet), `"gan"`, `"vrt"`
- **Dataset type** (`data/select_dataset.py`): `"sr"`, `"dncnn"`, `"ffdnet"`, `"fdncnn"`, `"dpsr"`, `"srmd"`, `"blindsr"`, `"jpeg"`, `"usrnet"`, plus video datasets
- **Network type** (`models/select_network.py`): `net_type` in `netG` config section, e.g. `"msrresnet0"`, `"rrdb"`, `"dncnn"`, `"swinir"`, `"vrt"`, `"rvrt"`, `"usrnet"`, `"drunet"`, `"imdn"`

### Model hierarchy

`ModelBase` → `ModelPlain` (PSNR) / `ModelGAN` / `ModelVRT`

`ModelBase` handles: checkpoint save/load, BN merging, EMA update, DataParallel/DDP wrapping, scheduler stepping.
Training loop lifecycle: `feed_data` → `optimize_parameters` → periodic test/save/log.

## Important quirks

- **GAN multi-GPU**: For GAN training with multiple GPUs, `model_gan.py:105` (the `DataParallel` wrapping of VGG) may cause bugs. The README recommends removing or commenting it.
- **Window-based models (SwinIR, VRT, RVRT)**: Input spatial/temporal dimensions must be multiples of `window_size`. Test scripts auto-pad with reflection padding (flip) and crop results back.
- **VRT/RVRT tiling**: Use `--tile` to split video spatially/temporally for GPU memory. First arg is temporal clip size, second two are spatial patch sizes. `[0,0,0]` = no tiling.
- **BN merging**: If a network has BN, the config's `merge_bn` + `merge_bn_startpoint` controls merging BN layers into preceding convs during training. After merging, use `act_mode='R'` (not `'BR'`) for testing.
- **Checkpoint files**: `{iteration}_G.pth`, `{iteration}_E.pth` (EMA), `{iteration}_D.pth` (discriminator), `{iteration}_optimizerG.pth`, `{iteration}_optimizerD.pth`. The training script scans filenames with `re.findall(r"(\d+)_{net_type}.pth", file_)`.
- **SwinIR/SwinIR3D**: Use `act_mode='R'` (no BN). SwinIR-3D models go under `options/swinir3d/` (not in this repo; external addition).
- **Test scripts auto-download**: SwinIR, VRT, and RVRT test scripts download models and datasets automatically if not found locally.
- **No test framework**: There are no pytest/unittest files. Testing means running inference scripts and checking PSNR/SSIM.
- **OrderedDict used everywhere** for determinism in configs and output dicts.
- **Training loop is infinite** (`for epoch in range(1000000)`). Users manually stop when converged.
- **`requirement.txt`** (singular) not `requirements.txt`.
