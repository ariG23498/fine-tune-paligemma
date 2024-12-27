# Object Detection with PaliGemma

This folder contains code for creating dataset and fine-tuning the PaliGemma vision-language model for license plate detection.

## Overview

The folder consists of two main components:
1. Dataset preparation `create_od_dataset.py`
2. Model fine-tuning and inference `object_detection_ft.py`, and also an interactive notebook `license_plate_detection_paligemma_ft.ipynb`

## Dataset

The project uses the license plate detection dataset from Hugging Face Hub:
- Source dataset: [`keremberke/license-plate-object-detection`](https://huggingface.co/datasets/keremberke/license-plate-object-detection)
- Processed dataset: (`license-detection-paligemma`)
The processed dataset is pushed to hub.

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

## Configuration

Key parameters can be modified in `configs/object_detection_config.py`:
- BATCH_SIZE
- LEARNING_RATE
- EPOCHS
- MODEL_ID
- DATASET_ID

