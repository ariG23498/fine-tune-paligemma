import torch
import argparse
from transformers import (
    AutoProcessor, 
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image


# Add argument parser
parser = argparse.ArgumentParser(description='Fine-tune PaLI-Gemma model')
parser.add_argument('--model_id', type=str, default="google/paligemma-3b-pt-224",
                   help='Model ID from Hugging Face Hub (default: google/paligemma-3b-pt-224)')
parser.add_argument('--output_dir', type=str, default="paligemma_intersections",
                   help='Directory to save the model outputs (default: paligemma_intersections)')
args = parser.parse_args()

# Configuration
dataset_id = "ariG23498/intersection-dataset"
model_id = args.model_id
device = "cuda:0"
dtype = torch.bfloat16
batch_size = 8

# Load dataset
train_ds = load_dataset(dataset_id, split='train')
val_ds = load_dataset(dataset_id, split='validation')

# Model configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Initialize model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map=device)
model = get_peft_model(model, lora_config)
processor = AutoProcessor.from_pretrained(model_id)

# Data collation function
def collate_fn(examples):
    images = [example["image"].convert("RGB") for example in examples]
    prompt = ["<image> How many intersections are there between the lines" for _ in examples]
    suffix = [str(example['label']) for example in examples]
    
    inputs = processor(
        images=images,
        text=prompt,
        suffix=suffix,
        return_tensors='pt',
        padding='longest'
    )
    
    inputs = inputs.to(dtype).to(device)
    return inputs

# Training configuration
training_args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir=args.output_dir,
    bf16=True,
    report_to=["wandb"],
    dataloader_pin_memory=False,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=training_args
)

# Start training
trainer.train()

#push the model to the huggingface hub
trainer.push_to_hub()
