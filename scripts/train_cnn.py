'''
TextCNN
'''
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import warnings
from tqdm import tqdm
import random
import json
import argparse

warnings.filterwarnings('ignore')


class Config:
    output_dir = "output"
    seeds = [42, 123, 456, 789, 999]
    max_len_options = [50, 100, 150]
    embed_dim_options = [100, 200, 300]
    filter_sizes_options = [[2, 3, 4], [3, 4, 5], [2, 4, 6]]
    num_filters_options = [50, 100, 150]
    dropout_options = [0.3, 0.4, 0.5]
    batch_size_options = [32, 64, 128]
    lr_options = [1e-3, 2e-3, 5e-3]
    epoch_options = [10, 15, 20]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        conv_out = [torch.max_pool1d(torch.relu(conv(embedded)).squeeze(3), conv(embedded).size(3)).squeeze(2) for conv in self.convs]
        cat = self.dropout(torch.cat(conv_out, dim=1))
        return self.fc(cat)


class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = str(self.texts[idx]).split()
        indices = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def build_vocab(texts, max_features=20000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(texts)
    vocab = vectorizer.get_feature_names_out()
    word2idx = {word: i + 2 for i, word in enumerate(vocab)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    return word2idx, len(word2idx)


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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        loss = criterion(model(texts), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(model(texts), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    wf1 = f1_score(all_labels, all_preds, average='weighted')
    mf1 = f1_score(all_labels, all_preds, average='macro')
    mae = mean_absolute_error(all_labels, all_preds)
    return total_loss / len(loader), acc, wf1, mf1, mae, all_preds


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(seed, train_data, dev_data, test_data, test_df, word2idx, vocab_size, device):
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = train_data, dev_data, test_data
    set_seed(seed)

    max_len = np.random.choice(Config.max_len_options)
    embed_dim = np.random.choice(Config.embed_dim_options)
    filter_sizes = np.random.choice(Config.filter_sizes_options)
    num_filters = np.random.choice(Config.num_filters_options)
    dropout = np.random.choice(Config.dropout_options)
    batch_size = np.random.choice(Config.batch_size_options)
    lr = np.random.choice(Config.lr_options)
    epochs = np.random.choice(Config.epoch_options)

    train_loader = DataLoader(TextDataset(X_train, y_train, word2idx, max_len), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TextDataset(X_dev, y_dev, word2idx, max_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TextDataset(X_test, y_test, word2idx, max_len), batch_size=batch_size, shuffle=False)

    model = TextCNN(vocab_size, embed_dim, 4, filter_sizes, num_filters, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None
    for _ in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer, device)
        _, acc, _, _, _, _ = evaluate(model, dev_loader, criterion, device)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    _, _, wf1, mf1, mae, preds = evaluate(model, test_loader, criterion, device)
    return {'wf1': wf1, 'mf1': mf1, 'mae': mae, 'preds': preds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    train_data, dev_data, test_data, test_df = load_data(args.data_dir)
    all_text = train_data[0] + dev_data[0] + test_data[0]
    word2idx, vocab_size = build_vocab(all_text)

    results = []
    for seed in Config.seeds:
        print(f"\n=== Seed: {seed} ===")
        res = run_experiment(seed, train_data, dev_data, test_data, test_df, word2idx, vocab_size, device)
        results.append(res)

    print("\n=== Results ===")
    for metric in ['wf1', 'mf1', 'mae']:
        vals = [r[metric] for r in results]
        print(f"Test {metric.upper()}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == '__main__':
    main()