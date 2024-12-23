# Intersection Dataset and Fine-Tuning PaliGemma 2 Base

## Overview
This project stems from an exploration of Lucas Beyer's [blog post on big_vision](https://lucasb.eyer.be/articles/bv_tuto.html), particularly the task of "counting intersection." Counting intersection involves identifying and counting the number of intersections formed when two lines cross each other within an image. The goal was to demonstrate how to:

1. Build a dataset for the task.
2. Upload it to the Hugging Face Hub for easy access.
3. Fine-tune a small `PaliGemma 2 base` model on this dataset.
4. Upload the fine-tuned model to Hugging Face Hub as well.

This repository also welcomes beginners in open-source contributions. There are ample opportunities to optimize, modularize, and enhance the codebase. Feel free to open PRs, raise issues, and review contributions.

## Results

| 0 Data Points | 20 Data Points | 100 Data Points | 200 Data Points |
|:--:|:--:|:--:|:--:|
|![image](https://github.com/user-attachments/assets/f62a9f5b-525c-4c07-8ecb-94a4b89d966a)|![image](https://github.com/user-attachments/assets/78f96d7b-ec1b-4a1e-92ba-dc42d7364dc0)|![image](https://github.com/user-attachments/assets/62e848b1-72e5-42a8-a080-1e8c2ec890e3)|![image](https://github.com/user-attachments/assets/18a17526-b405-4c0b-9007-689f0dba8cc5)|

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
    --push_to_hub=True \
    --dataset_id="YOUR_HF_USERNAME/intersection-dataset"
```

- **Parameters**:
  - `image_size`: Size of the generated images.
  - `dpi`: DPI setting for image clarity.
  - `num_images`: Number of images per intersection label.
  - `dataset_folder`: Directory to save the dataset.
  - `dataset_split`: Dataset split (`train` or `validation`).
  - `push_to_hub`: Whether to upload the dataset to the Hugging Face Hub.
  - `dataset_id`: ID used to upload dataset to the Hub, e.g. `ariG23498/intersection-dataset`. Required if `push_to_hub` is `True`.

The uploaded dataset is available [here](https://huggingface.co/datasets/ariG23498/intersection-dataset).

### Model Fine-Tuning

Fine-tune the `PaliGemma 2 base` model using:

```bash
python finetune.py \
    --ckpt_id="google/paligemma2-3b-pt-224" \
    --push_to_hub=True \
    --model_id="YOUR_HF_USERNAME/count_intersection-ft-paligemma2-3b-pt-224"
```

- **Parameters**:
  - `ckpt-id`: Pre-trained model checkpoint ID. Defaults to `google/paligemma2-3b-pt-224`.
  - `dataset_folder`: Directory containing the dataset, or dataset ID from Hugging Face. Defaults to `dataset`. 
  - `push_to_hub`: Whether to upload the fine-tuned model to the Hugging Face Hub.
  - `model_id`: ID used to upload model to the Hub, e.g. `oliveirabruno01/count_intersection-ft-paligemma2-3b-pt-224`. Required if `push_to_hub` is `True`.

### Example Workflow
1. You may need to authenticate with `huggingface-cli login` if you haven't already. 
2. Generate the dataset using `create_intersection_dataset.py`.
3. Fine-tune the model using `finetune.py`.
4. Run inference to visualize predictions and validate model performance.
