from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, src_data, trg_data, min_len, max_len):
        self.src_data = src_data
        self.trg_data = trg_data
        self.min_len = min_len
        self.max_len = max_len

        self.padded_data = []

        padding = [0 for i in range(self.max_len)]
        self.src_padded_data = []
        self.trg_padded_data = []
        self.total_padded_data = []

        for i in self.src_data:
            if self.min_len < len(i) < self.max_len:
                self.src_padded_data.append(i + padding[len(i):])
        self.total_padded_data.append(self.src_padded_data)

        for i in self.trg_data:
            if self.min_len < len(i) < self.max_len:
                self.trg_padded_data.append(i + padding[len(i):])
        self.total_padded_data.append(self.trg_padded_data)

        self.total_padded_data = tuple(self.total_padded_data)

    def __getitem__(self, data_type, idx):
        if data_type == "src":
            return self.total_padded_data[0][idx]
        elif data_type == "trg":
            return self.total_padded_data[1][idx]