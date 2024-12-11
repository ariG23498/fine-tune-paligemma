## Finetune PaliGemma2-448 variant

Finetuning script for the 448 variant of Paligemma2. I used LoRA adapters to fine-tune the entire model without freezing any blocks. 

The checkpoint can be found here: https://huggingface.co/LuciexJune/Paligemma2_lora

## Results:

![image](https://github.com/user-attachments/assets/ae724e87-ef8a-4d4b-9de2-24687cad328b)

Seems like to work maybe additional tuning might be needed to improve the performance.

Training args:
```python
rgs=TrainingArguments(
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf", # you can use paged optimizers like paged_adamw_8bit for QLoRA
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            output_dir="paligemma_intersections",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )
```
