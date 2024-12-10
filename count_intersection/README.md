# Intersection Dataset and Fine-Tuning PaliGemma 2 Base



## Introduction

This project explores the challenge of counting intersections in images using Vision-Language Models (VLMs). Inspired by the *["VLMs are Blind"](https://arxiv.org/abs/2305.12345)* paper and Lucas Beyer’s blog post on low-level vision tasks, the project aims to:

- **Build a dataset** tailored for the task of counting intersections.
- **Fine-tune Google's `PaliGemma 2` model** to perform this task effectively.
- **Encourage open-source contributions** by providing a beginner-friendly codebase.

The project not only demonstrates how to leverage state-of-the-art VLMs for specific low-level vision tasks but also creates opportunities for contributors to improve and extend the work.


## Background

### BlindTest Benchmark
**BlindTest**, a benchmark designed to evaluate Vision-Language Models (VLMs) on simple low-level visual tasks. These tasks, such as identifying geometric primitives (e.g., lines, circles, intersections), are fundamental to many image-related applications.

#### Key Highlights from the Paper:
- **Purpose**: Test VLMs' ability to “see” and process basic visual primitives rather than relying on memorization or pre-existing world knowledge.  
- **Tasks**: Seven novel tasks involving simple geometric operations like checking intersections or relationships between shapes.  
- **Surprising Findings**: Current state-of-the-art VLMs struggle with these simple tasks, sometimes failing to match the cognitive abilities of a five-year-old.  
- **Challenges for VLMs**: Poor performance is attributed to the **late-fusion** approach of integrating vision into language models, indicating a need for **early-fusion** techniques for better comprehension of visual prompts.  
- **Relevance to Real-World Tasks**: VLMs' inability to recognize low-level details could impact their performance in real-world scenarios, like following arrow directions or understanding maps.

### Lucas Beyer’s Blog Post
This project also draws from [Lucas Beyer's](https://lucasb.eyer.be/articles/bv_tuto.html) insightful blog post on low-level vision tasks, emphasizing the challenges and opportunities for improving VLMs in understanding geometric relationships.

#### Relevance of the Blog:
- It bridges the gap between the research findings in *"VLMs are Blind"* and practical approaches for tackling these challenges.  
- Inspired the creation of this project's dataset and methodology for fine-tuning a model to count intersections, a specific BlindTest task.

## Features

This project provides a comprehensive workflow for testing Vision-Language Models (VLMs) on the task of counting intersections in images. The key features include:

- **Dataset Creation**: 
  - Generate synthetic images that contain various geometric shapes, including lines and intersections. The dataset is labeled with the number of intersections (0, 1, or 2).
  - The dataset is designed to test VLMs on tasks that are simple for humans but challenging for current VLMs.

- **Dataset Hosting**:
  - The dataset is uploaded to the Hugging Face Hub, making it easily accessible for further use or fine-tuning on other models.
  - You can directly download or use the dataset in your projects with a simple call to the Hugging Face Hub.

- **Model Fine-Tuning**:
  - Fine-tune the `PaliGemma 2` base model for the intersection counting task using the generated dataset.
  - The model is trained to predict the number of intersections in a given image, leveraging the synthetic dataset.

- **Beginner-Friendly**:
  - The codebase is designed to be beginner-friendly, making it a great starting point for anyone looking to contribute to open-source projects.
  - Opportunities for improvement and extension, including refining the README, modularizing the code, or adding new tasks.

These features provide a solid foundation for working with VLMs, making the project both a learning experience and a valuable contribution to the open-source community.


## Dataset

The dataset for this project consists of synthetic images containing geometric shapes (such as lines, circles, and squares) with labeled intersection counts (0, 1, or 2). This dataset is designed to challenge Vision-Language Models (VLMs) in counting the number of intersections formed when two lines cross each other in an image.

### Key Characteristics:
- **Image Content**: Each image contains different geometric primitives (lines, circles, etc.) arranged in various ways to create intersections.
- **Labels**: Each image is labeled with the number of intersections (0, 1, or 2).
- **Synthetic Generation**: The dataset is synthetically generated to ensure diversity and coverage of different intersection scenarios.
- **Task Focus**: The dataset is specifically created to test the ability of VLMs to count intersections, which is part of a broader benchmark for evaluating the models' performance on basic visual tasks.

### Dataset Creation

To generate the dataset, run the following command:

```bash
python create_intersection_dataset.py \
    --image_size=224 \
    --dpi=100 \
    --num_images=100 \
    --dataset_folder="dataset" \
    --dataset_split="train" \
    --push_to_hub=True
```


## Model Fine-Tuning

In this project, we fine-tune the `PaliGemma 2` base model on the intersection counting task using the synthetic dataset. Fine-tuning a pre-trained model allows us to adapt the model’s capabilities to the specific task of counting intersections in images, leveraging transfer learning for better performance.

### Steps for Fine-Tuning

To fine-tune the `PaliGemma 2` model on the intersection dataset, follow these steps:

1. **Clone the repository** and navigate to the project directory.
2. **Ensure the dataset** is generated and accessible (see the [Dataset](#dataset) section).
3. **Run the fine-tuning script** with the following command:

```bash
python finetune.py \
    --ckpt-id="google/paligemma2-3b-pt-224" \
    --dataset_folder="path/to/your/dataset"
```


## Getting Started

To get started with this project, follow the steps below to set up the environment, generate the dataset, and fine-tune the model. These instructions will help you replicate the task of counting intersections in images using Vision-Language Models (VLMs).

### Prerequisites

Before you begin, ensure that you have the following installed:
- **Python** (version 3.7 or higher)
- **pip** (for installing Python packages)
- **Git** (for cloning the repository)
- **Hugging Face account** (for accessing and uploading datasets)

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/intersection-counting.git
cd intersection-counting
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Generate the Dataset

The dataset is synthetically generated using a script that creates images with labeled intersections. To generate the dataset, run the following command:
```
python create_intersection_dataset.py \
    --image_size=224 \
    --dpi=100 \
    --num_images=100 \
    --dataset_folder="dataset" \
    --dataset_split="train" \
    --push_to_hub=True
```
Parameters:
- image_size: Defines the size of the generated images (224 pixels by default).
- dpi: Adjusts the DPI (dots per inch) for image clarity.
- num_images: Specifies the number of images per intersection label (0, 1, or 2).
- dataset_folder: The directory where the dataset will be saved.
- dataset_split: Whether the dataset is for training or validation.
- push_to_hub: If set to True, the dataset will be uploaded to the Hugging Face Hub.


### 4. Finetune the model

Once the dataset is generated, fine-tune the pre-trained PaliGemma 2 model on the intersection counting task:

```
python finetune.py \
    --ckpt-id="google/paligemma2-3b-pt-224" \
    --dataset_folder="dataset"
```
- ckpt-id: The pre-trained model checkpoint ID. Here, we are using google/paligemma2-3b-pt-224.
- dataset_folder: The directory containing the generated dataset.


### 5. Running inference

```
python inference.py
```


### Workflow
1. Generate the dataset using `create_intersection_dataset.py`.
2. Fine-tune the model using `finetune.py`.
3. Run inference to visualize predictions and validate model performance.


### 6. Contributions

If you're interested in contributing to the project, you can:

- Improve the README.
- Add new features or optimizations to the codebase.
- Report issues or suggest enhancements.

To get started:
1. Fork the repository.
2. Clone your fork.
3. Create a new branch for your changes.
4. Commit your changes and push them to your fork.
5. Open a pull request with a detailed description of your changes.

Feel free to fork the repository, submit pull requests, and participate in discussions!
Let's build something amazing together! 🙌





