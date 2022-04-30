import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, src_data, trg_data, min_len, max_len):
        self.src_data = src_data
        self.trg_data = trg_data
        self.min_len = min_len
        self.max_len = max_len

        self.total_padded_data = []

        for src, trg in zip(self.src_data, self.trg_data):
            if self.min_len < len(src) < self.max_len:
                self.src_padded_data = torch.zeros(self.max_len, dtype=torch.long)
                self.src_padded_data[:len(src)] = torch.LongTensor(src)
            if self.min_len < len(trg) < self.max_len:
                self.trg_padded_data = torch.zeros(self.max_len, dtype=torch.long)
                self.trg_padded_data[:len(trg)] = torch.LongTensor(trg)

            self.total_padded_data.append((self.src_padded_data, self.trg_padded_data))

        self.total_padded_data = tuple(self.total_padded_data)

    def __getitem__(self, idx):
        return self.total_padded_data[idx]

    def __len__(self):
        return len(self.total_padded_data)