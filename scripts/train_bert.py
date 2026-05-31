'''
BERT-base (Chinese)
'''
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import warnings
from tqdm import tqdm
import random
import json
import argparse

warnings.filterwarnings('ignore')


class Config:
    bert_path = "lib/bert-base"
    output_dir = "output"
    seeds = [42, 123, 456, 789, 999]
    max_len_options = [64, 128, 256]
    batch_size_options = [8, 16, 32]
    lr_options = [2e-5, 3e-5, 5e-5]
    epoch_options = [3, 4, 5]
    dropout_options = [0.1, 0.2, 0.3]


class BertForSequenceClassification(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.1):
        super(BertForSequenceClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.bert_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average='weighted')
    mf1 = f1_score(all_labels, all_preds, average='macro')
    mae = mean_absolute_error(all_labels, all_preds)

    return avg_loss, acc, wf1, mf1, mae, all_preds


def load_data(data_dir):
    train_df = pd.read_excel(os.path.join(data_dir, 'train.xlsx'))
    dev_df = pd.read_excel(os.path.join(data_dir, 'dev.xlsx'))
    test_df = pd.read_excel(os.path.join(data_dir, 'test.xlsx'))

    X_train = train_df['sentence'].astype(str).tolist()
    y_train = train_df['label'].tolist()
    X_dev = dev_df['sentence'].astype(str).tolist()
    y_dev = dev_df['label'].tolist()
    X_test = test_df['sentence'].astype(str).tolist()
    y_test = test_df['label'].tolist()

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test), test_df


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(seed, train_data, dev_data, test_data, test_df, device):
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = train_data, dev_data, test_data

    set_seed(seed)

    max_len = random.choice(Config.max_len_options)
    batch_size = random.choice(Config.batch_size_options)
    lr = random.choice(Config.lr_options)
    num_epochs = random.choice(Config.epoch_options)
    dropout_rate = random.choice(Config.dropout_options)

    tokenizer = AutoTokenizer.from_pretrained(Config.bert_path)

    train_dataset = BertDataset(X_train, y_train, tokenizer, max_len)
    dev_dataset = BertDataset(X_dev, y_dev, tokenizer, max_len)
    test_dataset = BertDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification(num_classes=4, dropout_rate=dropout_rate).to(device)

    criterion = nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': lr},
    ]
    optimizer = AdamW(optimizer_grouped, lr=lr, eps=1e-8)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_dev_acc = 0
    best_state = None
    patience = 2
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        dev_loss, dev_acc, dev_wf1, dev_mf1, dev_mae, _ = evaluate(model, dev_loader, criterion, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, dev_acc, dev_wf1, dev_mf1, dev_mae, _ = evaluate(model, dev_loader, criterion, device)
    _, test_acc, test_wf1, test_mf1, test_mae, test_preds = evaluate(model, test_loader, criterion, device)

    exp_dir = os.path.join(Config.output_dir, f'bert_seed_{seed}')
    os.makedirs(exp_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        'id': test_df['id'],
        'sentence': test_df['sentence'],
        'true_label': y_test,
        'pred_label': test_preds
    })
    pred_df.to_excel(os.path.join(exp_dir, 'predictions.xlsx'), index=False)

    return {
        'dev_acc': dev_acc, 'dev_wf1': dev_wf1, 'dev_mf1': dev_mf1, 'dev_mae': dev_mae,
        'test_acc': test_acc, 'test_wf1': test_wf1, 'test_mf1': test_mf1, 'test_mae': test_mae
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dataset',
                        help='Directory containing train.xlsx, dev.xlsx, test.xlsx')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(Config.output_dir, exist_ok=True)

    # Check BERT files
    for f in ['pytorch_model.bin', 'config.json', 'vocab.txt']:
        if not os.path.exists(os.path.join(Config.bert_path, f)):
            raise FileNotFoundError(f"Missing {f} in {Config.bert_path}")

    train_data, dev_data, test_data, test_df = load_data(args.data_dir)

    all_results = []

    for seed in Config.seeds:
        print(f"\n=== Seed: {seed} ===")
        res = run_experiment(seed, train_data, dev_data, test_data, test_df, device)
        all_results.append(res)

    print("\n=== Results ===")
    for metric in ['wf1', 'mf1', 'acc', 'mae']:
        dev_vals = [r[f'dev_{metric}'] for r in all_results]
        test_vals = [r[f'test_{metric}'] for r in all_results]
        print(f"Dev {metric.upper()}: {np.mean(dev_vals):.4f} ± {np.std(dev_vals):.4f}")
        print(f"Test {metric.upper()}: {np.mean(test_vals):.4f} ± {np.std(test_vals):.4f}")


if __name__ == "__main__":
    main()