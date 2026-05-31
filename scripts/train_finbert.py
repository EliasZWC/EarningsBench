'''
FinBERT (Chinese)
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
import random
import json
import argparse

warnings.filterwarnings('ignore')


class Config:
    bert_path = "lib/finbert"
    output_dir = "output"
    seeds = [42, 123, 456, 789, 999]
    max_len_options = [128, 256, 384]
    batch_size_options = [4, 8, 16]
    lr_options = [1e-5, 2e-5, 3e-5]
    epoch_options = [3, 4, 5]
    dropout_options = [0.1, 0.2, 0.3]


class FinBERTForSequenceClassification(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.1):
        super(FinBERTForSequenceClassification, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained(Config.bert_path)
        except:
            self.bert = AutoModel.from_pretrained("ProsusAI/finbert")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class FinBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
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


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(seed, train_data, dev_data, test_data, test_df, device):
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = train_data, dev_data, test_data
    set_seed(seed)

    max_len = np.random.choice(Config.max_len_options)
    batch_size = np.random.choice(Config.batch_size_options)
    lr = np.random.choice(Config.lr_options)
    epochs = np.random.choice(Config.epoch_options)
    dropout = np.random.choice(Config.dropout_options)

    tokenizer = AutoTokenizer.from_pretrained(Config.bert_path)

    train_dataset = FinBERTDataset(X_train, y_train, tokenizer, max_len)
    dev_dataset = FinBERTDataset(X_dev, y_dev, tokenizer, max_len)
    test_dataset = FinBERTDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = FinBERTForSequenceClassification(num_classes=4, dropout_rate=dropout).to(device)

    criterion = nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and 'classifier' not in n],
         'weight_decay': 0.01, 'lr': lr * 0.1},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and 'classifier' not in n],
         'weight_decay': 0.0, 'lr': lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n],
         'weight_decay': 0.01, 'lr': lr},
    ]
    optimizer = AdamW(optimizer_grouped, lr=lr, eps=1e-8)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_acc = 0
    best_state = None
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        _, dev_acc, _, _, _, _ = evaluate(model, dev_loader, criterion, device)

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, _, wf1, mf1, mae, preds = evaluate(model, test_loader, criterion, device)

    exp_dir = os.path.join(Config.output_dir, f'finbert_seed_{seed}')
    os.makedirs(exp_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        'id': test_df['id'],
        'sentence': test_df['sentence'],
        'true_label': y_test,
        'pred_label': preds
    })
    pred_df.to_excel(os.path.join(exp_dir, 'predictions.xlsx'), index=False)

    return {'wf1': wf1, 'mf1': mf1, 'mae': mae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(Config.output_dir, exist_ok=True)

    train_data, dev_data, test_data, test_df = load_data(args.data_dir)

    results = []
    for seed in Config.seeds:
        print(f"\n=== Seed: {seed} ===")
        res = run_experiment(seed, train_data, dev_data, test_data, test_df, device)
        results.append(res)

    print("\n=== Results ===")
    for metric in ['wf1', 'mf1', 'mae']:
        vals = [r[metric] for r in results]
        print(f"Test {metric.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()