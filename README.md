# Cross-Modal Video Representation Alignment (CMVRA)

This repository contains the code for the paper **"Learning Robust Aligned Representations Across Multiple Visual Modalities in Human Action Recognition"**.

The project introduces the **Cross-Modal Video Representation Alignment (CMVRA)** framework, which aligns representations from diverse visual modalities (RGB, depth, IR, skeleton) using contrastive learning techniques. The framework builds upon CLIP-ViP and introduces novel multi-modal alignment losses.

## Overview

### Key Features

- **Multi-Modal Alignment**: Align representations from RGB, depth, IR, and skeleton modalities
- **Novel Loss Functions**: Implements MM-SWNCE (Multi-Modal Soft-Weighted NCE) with:
  - Weighting for faulty positive robustness
  - Cycle-consistency for faulty negative handling
  - Self-similarity for intra-modal consistency
- **Pretrained Models**: CLIP-ViP and MAE encoder checkpoints available
- **Task Support**:
  - Video retrieval with text encoder
  - Multi-modal alignment training
  - Classification with aligned features
  - MAE training and evaluation

### Paper Information

**Title**: Learning Robust Aligned Representations Across Multiple Visual Modalities in Human Action Recognition  
**Type**: Conference Paper  
**Authors**: David J. Lerch, Livien Majer, Zeyun Zhong, Manuel Martin, Frederik Diederichs, Rainer Stiefelhagen  
**Year**: 2026  
**Preprint**: https://arxiv.org/abs/2606.02352

### Important Notes for Users

⚠️ **Before running this code, you MUST**:

1. **Install requirements**: `pip install -r requirements.txt`
2. **Set up environment variables**: Copy `.env.example` to `.env` and fill in your NTU credentials
3. **Update hardcoded paths**: Many preprocessing scripts contain hardcoded paths like `/home/bas06400/` and `/net/polaris/` that must be replaced with your actual data locations
4. **Download datasets**: Ensure you have access to NTU RGB+D and DAA datasets
5. **Hardware requirements**: Minimum 16GB VRAM for training

**To find hardcoded paths in your codebase**:
```bash
grep -r "/home/bas06400\|/net/polaris" Dataset_utils/
```

**To update paths**: Edit the relevant lines in preprocessing scripts with your actual paths.

### Citation

If you use this code or our results in your research, please cite:

```bibtex
@article{lerch2026cmvra,
  title   = {Learning Robust Aligned Representations Across Multiple Visual Modalities in Human Action Recognition},
  author  = {David J. Lerch and Livien Majer and Zeyun Zhong and Manuel Martin and Frederik Diederichs and Rainer Stiefelhagen},
  journal = {arXiv preprint arXiv:2606.02352},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.02352}
}
```

**Preprint**: https://arxiv.org/abs/2606.02352

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA-enabled GPU (recommended for training)
- pip package manager

### Environment Variables

Set up the `.env` file for sensitive data:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual values
nano .env  # or your preferred editor
```

| Variable | Description | Required |
|----------|-------------|----------|
| `NTU_USERNAME` | NTU dataset login username | Yes (for NTU download) |
| `NTU_PASSWORD` | NTU dataset login password | Yes (for NTU download) |
| `NTU_DOWNLOAD_DIR` | Directory for downloaded data | No (default: `/net/polaris/storage/deeplearning/ntu`) |

⚠️ **Never commit** `.env` to version control!

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA-enabled GPU (recommended for training)
- pip package manager

### Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/LivienMajer/cmvra.git
   cd cmvra
   ```

2. **Create a virtual environment** (recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (for NTU dataset download):
   ```sh
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your NTU credentials
   nano .env  # or your preferred editor
   ```

## Quick Start

### Basic smoke test (CPU-only)

This tests the basic imports and data loading without GPU requirements:

```python
import torch
from zeta.data_loader import load_dataloaders
from zeta.model_init import initialize_vip_encoder

# Test basic imports
print("✓ Imports successful")

# Check PyTorch version
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
```

### Minimal training example

```bash
# Configure your paths in a JSON config file (see Configuration section)
python VIP/src/main.py --config VIP/src/configs/examples/example_training.json
```

## Data Preparation

### Required Datasets

#### 1. NTU RGB+D Dataset

The NTU RGB+D dataset contains RGB videos, depth maps, skeleton data, and IR videos.

**Download**:
- Use the provided script: `Dataset_utils/NTU/downloadntu.py`
- Or download manually from [NTU官网](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- Requires registration and login credentials

**Dataset structure**:
```
nturgb+d_rgb/          # RGB videos (.avi)
nturgb+d_depth_masked/ # Depth maps (images)
nturgb+d_skeletons_npy/# Skeleton data (.npy)
nturgb+d_ir/          # IR videos (.avi)
```

#### 2. DAA (Daily and Ambient Activities) Dataset

Contains multi-camera RGB, IR, depth, and skeleton data for daily activities.

**Download**: Contact dataset authors for access

**⚠️ IMPORTANT**: The DAA dataset preprocessing scripts contain hardcoded paths. After downloading, you must:

1. Update file paths in `Dataset_utils/DAA/combined_data.py`
2. Update input/output paths in `Dataset_utils/DAA/*.py` scripts
3. Set your data directory paths in `Dataset_utils/DAA/extract_data.py`

**To find all hardcoded paths in the codebase**:
```bash
grep -r "/home/bas06400\|/net/polaris" Dataset_utils/
```

**Structure**:
```
kinect_color/   # Kinect RGB videos
kinect_ir/      # Kinect IR videos
kinect_depth_mp4/ # Kinect depth videos
openpose_3d/    # OpenPose 3D skeleton data
ceiling/        # Ceiling camera views
...
```

### Data Preprocessing

⚠️ **IMPORTANT**: Run preprocessing scripts in `Dataset_utils/` **after** updating all hardcoded paths:

```bash
# NTU preprocessing
python Dataset_utils/NTU/crop_low_res_videos.py
python Dataset_utils/NTU/clean_low_res.py

# DAA preprocessing (update paths first!)
python Dataset_utils/DAA/extract_data.py
python Dataset_utils/DAA/balancer2.py
```

**Quick search for hardcoded paths in your codebase**:
```bash
grep -r "/home/bas06400\|/net/polaris" Dataset_utils/
```

## Training

### Fine-tuning CLIP-ViP Models

#### Video Retrieval (RGB+Text)

```bash
python VIP/src/run_video_retrieval.py \
  --config VIP/src/configs/examples/ntu_retrieval_example.json
```

#### Multi-Modal Alignment (RGB+Depth+IR+Skeleton)

```bash
python VIP/src/main.py \
  --config VIP/src/configs/AlignmentSleep.json
```

### Configuration File Format

Create a JSON config file with the following structure:

```json
{
  "task": "alignment",
  "modalities": ["rgb", "depth", "ir", "skeleton"],
  "dataset": "DAA",
  "split": "0",
  "encoder_model": "CLIP-ViP",
  "loss_config": {
    "loss_name": "MM_SWNCE",
    "temperature": 0.1,
    "use_weighting": true,
    "use_soft_targets": true,
    "use_self_similarity": true,
    "soft_mix": 0.5,
    "selfsim_mix": 0.5
  },
  "train_batch_size": 8,
  "num_train_epochs": 20,
  "learning_rate": 1e-5,
  "seed": 42
}
```

See `MM_SWNCE_HYPERPARAMETERS.md` for detailed hyperparameter documentation.

## Evaluation

### LEP Evaluation (Label Embedding Projection)

```bash
python VIP/src/main.py \
  --config VIP/src/configs/examples/lep_evaluation.json \
  --task 2
```

### Video Retrieval Evaluation

```bash
python VIP/src/run_video_retrieval.py \
  --config VIP/src/configs/examples/retrieval_eval.json
```

### KNN Evaluation

```bash
python VIP/src/zeta/eval_knn.py \
  --config VIP/src/configs/examples/knn_eval.json
```

## Reproducibility

### Random Seeds

All random seeds are set to `42` by default for reproducibility. To reproduce results:

```python
import torch
torch.manual_seed(42)
```

### Hardware Requirements

- **Training**: 
  - Minimum: 1 × NVIDIA GPU with 16GB VRAM
  - Recommended: 2 × NVIDIA A100 (80GB) or 4 × RTX 3090 (24GB)

- **Evaluation**:
  - Minimum: 1 × NVIDIA GPU with 8GB VRAM
  - CPU-only evaluation possible but significantly slower

### Expected Training Time

- NTU RGB+D (small split): ~4-6 hours on 2×V100
- DAA dataset: ~8-12 hours on 2×V100

### Troubleshooting

**Out of memory errors**: Reduce `train_batch_size` in config

**Data loading errors**: Verify dataset paths in config match actual locations

**CUDA errors**: Ensure PyTorch and CUDA versions are compatible

## Configuration

### Path Configuration

⚠️ **IMPORTANT**: Many preprocessing scripts and configuration files contain hardcoded paths that must be replaced with your actual data locations.

**Internal paths found in codebase** (must be updated):
- `/home/bas06400/` - User-specific path
- `/net/polaris/storage/` - Internal network path
- `/Thesis/` - Internal project path

**How to update paths**:

1. **For preprocessing scripts** (`Dataset_utils/`):
   ```python
   # Example: Dataset_utils/DAA/balancer2.py
   input_file = '/path/to/your/daa/daa_split_train2_full.txt'
   output_file = '/path/to/your/daa/daa_split_train2_full_balanced.txt'
   ```

2. **For config files** (create your own config):
   ```json
   {
     "rgb_path": "/path/to/your/rgb/data",
     "depth_path": "/path/to/your/depth/data",
     "skeleton_path": "/path/to/your/skeleton/data"
   }
   ```

3. **For evaluation data**:
   ```python
   # Update paths in evaluation scripts
   data_file = "/path/to/your/evaluation/set.txt"
   ```

**Recommended approach**: Create a config file with all paths at the top and reference them throughout your scripts.

### Hyperparameter Reference

All MM-SWNCE hyperparameters can be configured in the JSON config:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_name` | str | "MM_SWNCE" | Loss function (NCE, MM_SWNCE) |
| `temperature` | float | 0.1 | Temperature for softmax |
| `soft_mix` | float | 0.5 | Soft target mixing ratio |
| `selfsim_mix` | float | 0.5 | Self-similarity mixing ratio |
| `use_weighting` | bool | true | Enable faulty positive weighting |
| `use_soft_targets` | bool | true | Enable cycle-consistency |
| `use_self_similarity` | bool | true | Enable intra-modal consistency |

See `MM_SWNCE_HYPERPARAMETERS.md` for complete documentation.

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NTU_USERNAME` | NTU dataset login username | `john.doe` |
| `NTU_PASSWORD` | NTU dataset login password | `secret123` |
| `NTU_DOWNLOAD_DIR` | Download directory path | `/data/ntu` |

## Directory Structure

```
cmvra/
├── Dataset_utils/
│   ├── NTU/              # NTU dataset preprocessing
│   ├── DAA/              # DAA dataset preprocessing
│   ├── VIP_datasets/     # Dataset utilities
│   ├── caption_task_LLaVa/
│   └── captions_from_excel/
├── VIP/
│   ├── src/
│   │   ├── configs/      # Configuration files
│   │   ├── datasets/     # Dataset implementations
│   │   ├── modeling/     # Model architectures
│   │   ├── optimization/ # Optimizers and schedulers
│   │   ├── utils/        # Utility functions
│   │   └── zeta/         # Main training/evaluation code
│   └── LICENSE
├── checkpoints/          # Saved model checkpoints
├── align_checkpoints/    # Alignment-specific checkpoints
├── .env.example         # Environment variable template
├── LICENSE             # MIT License
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Dependencies

### Core Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (for GPU training)

### Key Libraries

- `transformers` (HuggingFace)
- `torchvision`
- `pytorchvideo`
- `decord` (video loading)
- `tqdm` (progress bars)
- `easydict` (config management)

### Full List

See `requirements.txt` for complete dependency list.

## License

This project contains code from multiple sources with different licenses:

- **Main code (this repository)**: MIT License (see [LICENSE](LICENSE))
- **CLIP-ViP code**: MIT License (see `VIP/LICENSE`)
- **Omnivore/OmniMAE**: CC-BY-NC 4.0 (see `VIP/src/modeling/LICENSE`)

**Important**: The omnivore component is licensed under CC-BY-NC 4.0. Contact the authors for commercial use permissions.

## Acknowledgements

This work builds upon and extends the following repositories:

- [CLIP-ViP](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP) - Jie Lei
- [Omnivore / OmniMAE](https://github.com/facebookresearch/omnivore) - Facebook Research

We thank the authors for making their code available.

## Contact

For questions or issues, please open an issue on GitHub or contact:

**Author**: Livien Majer  
**Institution**: Fraunhofer IOSB  
**Email**: livien.majer@iosb.fraunhofer.de

## Security Notes

- **Never commit** `.env` files with real credentials
- **Use** `.env.example` as a template
- **Rotate** any accidentally exposed credentials immediately
- **Review** third-party dependencies for security vulnerabilities

## Hardcoded Paths

This codebase contains hardcoded paths from the original development environment. Before running the code, **you must update these paths** to match your local setup:

### How to Find Hardcoded Paths

```bash
# Search for common hardcoded paths
grep -r "/home/bas06400\|/net/polaris\|/Thesis" Dataset_utils/ VIP/src/

# Look for file paths in config files
grep -r "path" VIP/src/configs/*.json 2>/dev/null | head -20
```

### Common Files with Hardcoded Paths

| File | Type of Paths | Action |
|------|--------------|--------|
| `Dataset_utils/DAA/*.py` | Input/output data paths | Replace with your data locations |
| `Dataset_utils/NTU/*.py` | NTU download/output paths | Update download directory |
| Config files in `VIP/src/configs/` | Data directories | Create your own config |

### Best Practice

Instead of editing files in-place, **create your own configuration**:

```bash
# Copy an example config
cp VIP/src/configs/examples/example.json VIP/src/configs/my_config.json

# Edit your copy with your paths
nano VIP/src/configs/my_config.json
```

## Future Work

- [ ] Add more pretrained model checkpoints
- [ ] Support additional datasets
- [ ] Extend to 3D skeleton representations
- [ ] Multi-language text encoder support
