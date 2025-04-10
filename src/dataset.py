import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter


class WikiTextDataset(Dataset):
    def __init__(self, data_path, max_seq_length=512):
        self.max_seq_length = max_seq_length
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = text.split()
        self.counter = Counter(tokens)
        self.vocab = sorted(self.counter, key=self.counter.get, reverse=True)
        # 添加未知词标记
        self.vocab.append('<UNK>')
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        # 处理未知词
        self.data = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in tokens]

    def __len__(self):
        return len(self.data) // self.max_seq_length

    def __getitem__(self, idx):
        start = idx * self.max_seq_length
        end = min(start + self.max_seq_length, len(self.data))
        input_ids = torch.tensor(self.data[start:end], dtype=torch.long)
        target_ids = torch.tensor(self.data[start + 1:end + 1], dtype=torch.long)
        return input_ids[:-1], target_ids[:-1]
