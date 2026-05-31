'''
BiLSTM
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
import random
import json
import argparse

warnings.filterwarnings('ignore')


class Config:
    output_dir = "output"
    seeds = [42, 123, 456, 789, 999]
    max_len_options = [80, 100, 120]
    embed_dim_options = [100, 150, 200]
    hidden_dim_options = [128, 192, 256]
    num_layers_options = [1, 2]
    dropout_options = [0.3, 0.4, 0.5]
    batch_size_options = [32, 64]
    lr_options = [1e-3, 2e-3, 5e-3]
    epoch_options = [15, 20]


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout_rate):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(hidden))


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    param_sets = [
        {'max_len': 100, 'embed_dim': 100, 'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.4, 'batch_size': 32, 'lr': 2e-3, 'epochs': 15},
        {'max_len': 80, 'embed_dim': 150, 'hidden_dim': 192, 'num_layers': 2, 'dropout': 0.3, 'batch_size': 64, 'lr': 1e-3, 'epochs': 20},
        {'max_len': 120, 'embed_dim': 200, 'hidden_dim': 256, 'num_layers': 1, 'dropout': 0.5, 'batch_size': 32, 'lr': 5e-3, 'epochs': 15},
        {'max_len': 100, 'embed_dim': 150, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.4, 'batch_size': 64, 'lr': 2e-3, 'epochs': 20},
        {'max_len': 80, 'embed_dim': 100, 'hidden_dim': 192, 'num_layers': 1, 'dropout': 0.3, 'batch_size': 32, 'lr': 1e-3, 'epochs': 15}
    ]
    p = param_sets[seed % len(param_sets)]

    train_loader = DataLoader(TextDataset(X_train, y_train, word2idx, p['max_len']), batch_size=p['batch_size'], shuffle=True)
    dev_loader = DataLoader(TextDataset(X_dev, y_dev, word2idx, p['max_len']), batch_size=p['batch_size'], shuffle=False)
    test_loader = DataLoader(TextDataset(X_test, y_test, word2idx, p['max_len']), batch_size=p['batch_size'], shuffle=False)

    model = BiLSTM(vocab_size, p['embed_dim'], p['hidden_dim'], p['num_layers'], 4, p['dropout']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc, best_state = 0, None
    patience_counter, patience = 0, 5

    for epoch in range(p['epochs']):
        train_epoch(model, train_loader, criterion, optimizer, device)
        _, acc, _, _, _, _ = evaluate(model, dev_loader, criterion, device)
        scheduler.step(acc)
        if acc > best_acc:
            best_acc, best_state = acc, model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    _, _, wf1, mf1, mae, preds = evaluate(model, test_loader, criterion, device)

    exp_dir = os.path.join(Config.output_dir, f'bilstm_seed_{seed}')
    os.makedirs(exp_dir, exist_ok=True)
    pd.DataFrame({
        'id': test_df['id'],
        'sentence': test_df['sentence'],
        'true_label': y_test,
        'pred_label': preds
    }).to_excel(os.path.join(exp_dir, 'predictions.xlsx'), index=False)

    return {'wf1': wf1, 'mf1': mf1, 'mae': mae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(Config.output_dir, exist_ok=True)

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


if __name__ == "__main__":
    main()