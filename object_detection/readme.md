# Object Detection with PaliGemma

This folder contains code for preparing the dataset and fine-tuning the PaliGemma vision-language model for license plate detection.

## Overview

The folder consists of the following main components:
1. **Dataset Preparation**: `create_od_dataset.py`
2. **Model Fine-Tuning and Inference**:
   - Script: `object_detection_ft.py`
   - Interactive notebook: `license_plate_detection_paligemma_ft.ipynb`

## Dataset

The project utilizes a license plate detection dataset from the Hugging Face Hub:
- **Source Dataset**: [`keremberke/license-plate-object-detection`](https://huggingface.co/datasets/keremberke/license-plate-object-detection)
- **Processed Dataset**: `license-detection-paligemma` (uploaded to the Hugging Face Hub)

The processed dataset is converted into a PaliGemma-compatible format.

## Usage

1. First, prepare the dataset:
```bash
python create_od_dataset.py
```
This script converts COCO format annotations to PaliGemma-compatible format.

2. Run the fine-tuning:
```bash
python object_detection_ft.py
```