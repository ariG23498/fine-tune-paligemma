# Image Captioning with PaliGemma

This project demonstrates fine-tuning the PaliGemma vision-language model for image captioning. 



## Project Structure

- **`vanilla_ft.py`**: Main script for fine-tuning and evaluating the PaliGemma model.
- **`vanilla_paligemma_ft.ipynb`**: Interactive notebook for experimenting with fine-tuning and inference.
- **`configs/vanilla_config.py`**: Configuration file for model parameters like dataset ID, batch size, learning rate, and number of epochs.



## Features

- **Dataset Loading**: Fetches a dataset directly from the Hugging Face Hub using `datasets`.
- **Fine-Tuning**: Customizes the pre-trained PaliGemma model for the image captioning task.
- **Inference**: Generates captions for test images and visualizes the results using Matplotlib.
- **Configurable Parameters**: Modify settings like batch size and learning rate through the configuration file.



## Dataset

The dataset is directly loaded from the Hugging Face Hub:
- **Dataset ID**: Specified in the configuration file (`configs/vanilla_config.py`).
- **Current Dataset**: [diffusers/tuxemon](https://huggingface.co/datasets/diffusers/tuxemon)

The dataset is split into training and testing subsets automatically during execution.



## How to Use

### Step 1: Prepare Environment
1. Install required libraries:
    ```bash
    pip install torch transformers datasets matplotlib
    ```
2. Ensure the configuration file (`configs/vanilla_config.py`) is correctly set up with the desired parameters.

### Step 2: Run the Script
1. Train and fine-tune the model:
    ```bash
    python image_captioning_task.py
    ```
2. During training, logs will display training and validation losses at regular intervals.

3. After training, the script will visualize test images with generated captions before and after fine-tuning.





## Interactive Notebook

For an interactive and step-by-step fine-tuning process, use the provided Jupyter notebook:
```bash
jupyter notebook vanilla_paligemma_ft.ipynb
