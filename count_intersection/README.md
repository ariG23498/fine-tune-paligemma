# Intersection Dataset and Fine-Tuning PaliGemma 2 Base

## Overview
This project stems from an exploration of Lucas Beyer's [blog post on big_vision](https://lucasb.eyer.be/articles/bv_tuto.html), particularly the task of "counting intersection." Counting intersection involves identifying and counting the number of intersections formed when two lines cross each other within an image. The goal was to demonstrate how to:

1. Build a dataset for the task.
2. Upload it to the Hugging Face Hub for easy access.
3. Fine-tune a small `PaliGemma 2 base` model on this dataset.

This repository also welcomes beginners in open-source contributions. There are ample opportunities to optimize, modularize, and enhance the codebase. Feel free to open PRs, raise issues, and review contributions.

## Features
1. **Dataset Creation**: Generate synthetic images with labeled intersection counts (0, 1, or 2 intersections).
2. **Dataset Hosting**: The dataset is uploaded to the Hugging Face Hub for accessibility.
3. **Model Fine-Tuning**: Fine-tune `PaliGemma 2 base` for the intersection counting task.
4. **Beginner-Friendly**: Code designed to encourage open-source contributions.

## Quickstart

### Dataset Creation
Run the following command to generate the dataset:

```bash
python create_intersection_dataset.py \
    --image_size=224 \
    --dpi=100 \
    --num_images=100 \
    --dataset_folder="dataset" \
    --dataset_split="train" \
    --push_to_hub=True
```

- **Parameters**:
  - `image_size`: Size of the generated images.
  - `dpi`: DPI setting for image clarity.
  - `num_images`: Number of images per intersection label.
  - `dataset_folder`: Directory to save the dataset.
  - `dataset_split`: Dataset split (`train` or `validation`).
  - `push_to_hub`: Whether to upload the dataset to the Hugging Face Hub.

The uploaded dataset is available [here](https://huggingface.co/datasets/ariG23498/intersection-dataset).

### Model Fine-Tuning
Fine-tune the `PaliGemma 2 base` model using:

```bash
python finetune.py \
    --ckpt-id="google/paligemma2-3b-pt-224"
```

- **Parameters**:
  - `ckpt-id`: Pre-trained model checkpoint ID.
  - `dataset_folder`: Directory containing the dataset.

Fine-tune the whole model(without freezing any layers) with LoRA:
```bash
python lora_finetune.py --model_id "google/paligemma2-3b-pt-224" --output_dir "your/output/directory"
```
Can be trained on Google Colab with A100 40GB GPU runtime.

### Example Workflow
1. Generate the dataset using `create_intersection_dataset.py`.
2. Fine-tune the model using `finetune.py`.
3. Run inference to visualize predictions and validate model performance.
