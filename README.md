# Multi-Modal Alignment and Evaluation Framework

This repository contains a framework for multi-modal alignment and evaluation, with a focus on video-text retrieval and masked autoencoder (MAE) training. It builds upon and modifies the CLIP-ViP pipeline for fine-tuning CLIP models on video data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Fine-tuning CLIP-ViP Models](#fine-tuning-clip-vip-models)
  - [Multi-Modal Alignments](#multi-modal-alignments)
  - [Evaluation](#evaluation)
  - [MAE Training and Evaluation](#mae-training-and-evaluation)
- [Directory Structure](#directory-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

This project provides tools and scripts for:

- Fine-tuning CLIP-ViP models
- Performing multi-modal alignments
- Evaluating alignments using Language-Encoded Prior (LEP) under various settings
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

### Fine-tuning CLIP-ViP Models

Example configurations for fine-tuning CLIP-ViP models can be found in the `/Thesis/VIP/src/configs` directory. Look for files with "retrieval" in the name.

### Multi-Modal Alignments

After fine-tuning, the CLIP-ViP models can be used for alignments. Example configurations are also available in the config directory.

### Evaluation

- **LEP Evaluation**: Use the provided configurations to evaluate alignments under different settings.
- **Retrieval Evaluation**: Evaluate retrievals using a text encoder with the appropriate config file.

### MAE Training and Evaluation

Configurations for MAE training and evaluation are provided in the config directory.

## Directory Structure

- `/Thesis/VIP/src/configs`: Contains example configuration files
- `/zeta`: Contains personal code for alignments and evaluations
- `/zlogs` and `/zalignmentlogs`: Contain logs of experiments
- `main.py`: Main script for running experiments
- `config`: Directory containing various configuration files


### Dataset File Structure

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

- CLIP-ViP by Jie Lei
- Omnivore

We are grateful to the authors for making their code available.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### CLIP-ViP License

MIT License