# Example usage:
# python finetune.py \
# --ckpt-id="google/paligemma2-3b-pt-224"

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import random
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from functools import partial
from fire import Fire
from tqdm import tqdm
from matplotlib import pyplot as plt


def collate_fn(examples, processor):
    images = list()
    prompt = list()
    suffix = list()
    for sample in examples:
        images.append(sample["image"].convert("RGB"))
        suffix.append(str(sample["label"]))
        prompt.append("count intersection")

    inputs = processor(
        images=images,
        text=prompt,
        suffix=suffix,
        return_tensors="pt",
    )
    inputs = inputs.to(torch.bfloat16)
    return inputs


def freeze_layers(model):
    for name, param in model.named_parameters():
        if "attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def run_inference(val_dataset, processor, model):
    # infer before training
    val_sample = random.choice(val_dataset)
    image = val_sample["image"].convert("RGB")
    inputs = processor(
        images=[image],
        text=["count intersection"],
        return_tensors="pt",
    ).to(torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        generation = model.generate(**inputs.to(model.device), max_new_tokens=10)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Pred: {decoded}")
    plt.show()


def main(ckpt_id: str = "google/paligemma2-3b-pt-224", dataset_folder: str = "dataset"):
    # load the dataset and the processor
    print(f"[INFO] Loading {ckpt_id} processor")
    processor = PaliGemmaProcessor.from_pretrained(ckpt_id)

    # load the dataset
    print(f"[INFO] Loading {dataset_folder} dataset")
    train_dataset = load_dataset(dataset_folder, split="train")
    val_dataset = load_dataset(dataset_folder, split="validation")

    # create data loader
    print(f"[INFO] Creating dataloader")
    partial_collate_fn = partial(collate_fn, processor=processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=partial_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=partial_collate_fn,
    )

    # load the model and optimizer
    print(f"[INFO] Loading {ckpt_id} model")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        ckpt_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # run inference before training
    run_inference(val_dataset, processor, model)

    model = freeze_layers(model)
    print(f"[INFO] Model loaded on {model.device}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
    )

    # Start Training
    accumulation_steps = 8
    for idx, batch in tqdm(enumerate(train_dataloader)):
        outputs = model(**batch.to(model.device))
        loss = outputs.loss / accumulation_steps
        if idx % 50 == 0:
            val_loss = 0.0
            with torch.no_grad():
                count = 0
                for val_batch in val_dataloader:
                    val_loss = val_loss + model(**val_batch.to(model.device)).loss
                    count = count + 1
                val_loss = val_loss / count
            print(
                f"Iter: {idx} Loss: {loss.item():.4f} Val Loss: {val_loss.item():.4f}"
            )
            run_inference(val_dataset, processor, model)

        loss.backward()
        optimizer.step()
        if idx % 8 == 0:
            optimizer.zero_grad()


if __name__ == "__main__":
    Fire(main)
