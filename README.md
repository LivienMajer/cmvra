# Cross-Modal Video Representation Alignment (CMVRA) for Human Action Recognition

This repository contains the code and resources for the Master's Thesis "Learning Robust Aligned Representations Across Multiple Visual Modalities in Human Action Recognition". The project introduces the Cross-Modal Video Representation Alignment (CMVRA) framework, which aligns representations from diverse visual modalities using contrastive learning techniques and builds upon video extensions of CLIP: CLIP-ViP.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Captions Using LLaVa](#generate-captions-using-llava)
  - [Fine-tuning CLIP-ViP Models](#fine-tuning-clip-vip-models)
  - [Multi-Modal Alignments](#multi-modal-alignments)
  - [Pretrained Model Checkpoints](#pretrained-model-checkpoints)
  - [Selected Model Checkpoints](#selected-model-checkpoints)
  - [Evaluation](#evaluation)
  - [MAE Training and Evaluation](#mae-training-and-evaluation)
- [Directory Structure](#directory-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

This project provides tools and scripts for:

- Fine-tuning CLIP-ViP models
- Performing multi-modal alignments
- Evaluating alignments using LEP under various settings, as well as a fusion evaluation 
- Evaluating retrievals using a text encoder
- Training and evaluating Masked Autoencoders (MAE)

## Installation

To set up the project environment:

1. Clone this repository:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

This will install all the necessary dependencies for the project.

## Usage

### Generate Captions Using LLaVa

Clone this repository and navigate to LLaVA folder
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
git pull
pip install -e .
pip install flash-attn --no-build-isolation --no-cache-dir

then place this file Thesis/Dataset_utils/caption_task_LLaVa/caption.pyunder /llava/serve/caption.py

and run it with this command, adjust the path to your dataset file:

python -m llava.serve.caption     --model-path liuhaotian/llava-v1.6-34b     --data-file /home/bas06400/Thesis/CV_training_set.txt     --load-8bit --device cuda:0

"""

### Fine-tuning CLIP-ViP Models

Example configurations for fine-tuning CLIP-ViP models can be found in the `/Thesis/VIP/src/configs` directory. Look for files with "retrieval" in the name.

### Multi-Modal Alignments

After fine-tuning, the CLIP-ViP models can be used for alignments. Example configurations are also available in the config directory.
Alignments can also be performed with the base version weights published on https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP
Weights can be downloaded here: https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_16.pt?sp=r&st=2023-03-16T05:02:05Z&se=2026-07-31T13:02:05Z&spr=https&sv=2021-12-02&sr=b&sig=XNd7fZSsUhW7eesL3hTfYUMiAvCCN3Bys2TadXlWzFU%3D
If the checkpoint is not avaiable anymore just ask for the weights. Just place it in your dir and set the config['e2e_weights_path']

### Pretrained Model Checkpoints
Omnivore Repo: https://github.com/facebookresearch/omnivore/tree/main/omnivore
Omnivore Download: https://dl.fbaipublicfiles.com/omnivore/models/swinB_In21k_checkpoint.torch
OmniMAE Repo: https://github.com/facebookresearch/omnivore/tree/main/omnimae
OmniMAE Download:https://dl.fbaipublicfiles.com/omnivore/omnimae_ckpts/vitb_pretrain.torch
TAHAR Download: https://hessenbox-a10.rz.uni-frankfurt.de/getlink/fiKhHcKpHmgvnNJpka6V1Z/


### Selected Model Checkpoints

https://hessenbox-a10.rz.uni-frankfurt.de/getlink/fi8z62gJJq5SdVn2gihaSC/


### Evaluation

- **LEP Evaluation**: Use the provided configurations to evaluate alignments under different settings.
- **Retrieval Evaluation**: Evaluate retrievals using a text encoder with the appropriate config file.

### MAE Training and Evaluation

Configurations for MAE training and evaluation are provided in the config directory.

## Directory Structure

- `/Thesis/VIP/src/configs`: Contains example configuration files
- `/zeta`: Contains personal code for alignments and evaluations
- `/zlogs` and `/zalignmentlogs`: Contain logs of experiments. All logs contain the corresponding config.
- `main.py`: Main script for running experiments
- `config`: Directory containing various configuration files


### Dataset File Structure

RGB, IR and depth should be .avi video files and a sequneces of images for masked depth on NTU. Skeleton should be saved as numpy array.
The file structure for the NTU RGB+D dataset and the DAA dataset can be inferred from the following examples:

#### NTU RGB+D Dataset

The NTU RGB+D dataset is organized under the nturgb+d_* directories, each representing a different modality. An example file structure is as follows:

nturgb+d_rgb/   
├── ...    
nturgb+d_ir/    
├── ...   
nturgb+d_depth_masked/    
├── ...   
nturgb+d_skeletons_npy/  
├── ...   


#### DAA Dataset

The DAA dataset is organized under directories named after the different camera views, splits and modalities. You can recreate it running this file:
/Thesis/Dataset_utils/DAA/extract_data.py
on the DAA dataset as downloaded. An examplestructure is as follows:

kinect_color/  
├── clips/test/...  
kinect_ir/  
├── clips/test/...  
kinect_depth_mp4/  
├── clips/test/...   
openpose_3d/  
├── clips/test/...  
ceiling/  
├── clips/test/...  
inner_mirror/  
├── clips/test/...  
a_column_co_driver/  
├── clips/test/...   
a_column_driver/   
├── clips/test/...  
steering_wheel/  
├── clips/test/...   

## Acknowledgements

This project builds upon the following works:

- CLIP-ViP by Jie Lei https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP
- Omnivore / OmniMAE by Facebook reasearch https://github.com/facebookresearch/omnivore

We are grateful to the authors for making their code available.

## Omnivore / OmniMAE License

Omnivore is released under the CC-BY-NC 4.0 license. See LICENSE for additional details. However the Swin Transformer implementation is additionally licensed under the Apache 2.0 license (see NOTICE for additional details). Detailed License file can be found here: /VIP/src/modeling/LICENSE

### CLIP-ViP License

MIT License

Copyright (c) 2021 Jie Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.