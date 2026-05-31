'''
LLM + FinSCoT
'''
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import gc
import warnings
import argparse
import re

warnings.filterwarnings("ignore")


class Config:
    BASE_DIR = "."
    DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")
    TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
    DEV_FILE = os.path.join(DATA_DIR, "dev.json")
    TEST_FILE = os.path.join(DATA_DIR, "test.json")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    MODEL_DIR = os.path.join(BASE_DIR, "lib", "llm", "qwen")

    USE_4BIT = True
    BATCH_SIZE = 8
    SEEDS = [42, 123, 456, 789, 999]
    EPOCHS = 5
    LR = 3e-4


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if not self.is_test and "reason" in item and item["reason"]:
            input_text = f"Analyze the sentiment. Provide reasoning first, then the label.\nText: {item['input']}\n"
            output_text = f"Reasoning: {item['reason']}\nAnswer: {item['output']}"
        else:
            input_text = f"Analyze the sentiment. Output only the number.\nText: {item['input']}\nLabel (0-3): "
            output_text = str(item['output'])

        full_text = input_text + output_text
        encoding = self.tokenizer(full_text, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()

        input_encoding = self.tokenizer(input_text, truncation=True, padding="max_length",
                                        max_length=self.max_length, return_tensors="pt")
        input_len = input_encoding["input_ids"].shape[1]
        if input_len < self.max_length:
            labels[:input_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }


def load_data(filepath, is_test=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    if isinstance(json_data, list):
        raw_data = json_data
    elif isinstance(json_data, dict) and "data" in json_data:
        raw_data = json_data["data"]
    else:
        raw_data = []

    data = []
    for i, item in enumerate(raw_data):
        if not isinstance(item, dict):
            continue
        output_str = str(item.get("output", "0"))
        label = int(output_str[0]) if output_str and output_str[0].isdigit() else 0

        data_item = {
            "id": i,
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": output_str,
            "label": label
        }
        if not is_test and "reason" in item:
            data_item["reason"] = str(item["reason"])

        data.append(data_item)
    return data


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def extract_label_from_cot_text(text):
    match = re.search(r'答案[：:]\s*(\d)', text)
    if match:
        return int(match.group(1))
    match = re.search(r'Answer[：:]\s*(\d)', text)
    if match:
        return int(match.group(1))
    numbers = re.findall(r'\b(\d)\b', text)
    if numbers:
        return int(numbers[-1])
    return 0


def evaluate(model, tokenizer, test_data, test_dataset, batch_size=4):
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    predictions = []

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            prompts = []
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(test_data))
            for i in range(batch_start, batch_end):
                item = test_data[i]
                prompts.append(f"Analyze the sentiment. Provide reasoning first, then the label.\nText: {item['input']}\n")

            encoded = tokenizer(prompts, truncation=True, padding=True, max_length=512, return_tensors="pt").to(model.device)

            outputs = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for i, output in enumerate(outputs):
                prompt_len = encoded["input_ids"][i].shape[0]
                pred_text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                predictions.append(extract_label_from_cot_text(pred_text))

    tokenizer.padding_side = original_padding

    true_labels = [item["label"] for item in test_data]

    wf1 = f1_score(true_labels, predictions, average='weighted')
    mf1 = f1_score(true_labels, predictions, average='macro')
    acc = accuracy_score(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)

    return {"WF1": wf1, "MF1": mf1, "Accuracy": acc, "MAE": mae}, predictions


def run_experiment(seed):
    set_seed(seed)

    train_data = load_data(Config.TRAIN_FILE, is_test=False)
    dev_data = load_data(Config.DEV_FILE, is_test=False)
    test_data = load_data(Config.TEST_FILE, is_test=True)

    base_model, tokenizer = load_model_and_tokenizer(Config.MODEL_DIR)

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                             task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base_model, lora_config)

    train_dataset = SentimentDataset(train_data, tokenizer, is_test=False)
    dev_dataset = SentimentDataset(dev_data, tokenizer, is_test=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(Config.OUTPUT_DIR, f"lora_cot_seed_{seed}"),
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=Config.LR,
        weight_decay=0.01,
        fp16=True,
        logging_steps=5,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    test_dataset = SentimentDataset(test_data, tokenizer, is_test=True)
    metrics, _ = evaluate(model, tokenizer, test_data, test_dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    parser.add_argument('--model_dir', type=str, default='lib/llm/qwen')
    args = parser.parse_args()

    Config.DATA_DIR = args.data_dir
    Config.TRAIN_FILE = os.path.join(Config.DATA_DIR, "train.json")
    Config.DEV_FILE = os.path.join(Config.DATA_DIR, "dev.json")
    Config.TEST_FILE = os.path.join(Config.DATA_DIR, "test.json")
    Config.MODEL_DIR = args.model_dir

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    for seed in Config.SEEDS:
        print(f"\n=== Seed: {seed} ===")
        metrics = run_experiment(seed)
        all_metrics.append(metrics)

    print("\n=== Results ===")
    for metric in ["WF1", "MF1", "Accuracy", "MAE"]:
        values = [m[metric] for m in all_metrics]
        print(f"{metric}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    main()