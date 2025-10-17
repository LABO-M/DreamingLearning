
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class TorchDataWindow(Dataset):
    def __init__(self, data, input_width, label_width, shift=1, stride=1, label_columns=None, feature_names=None):
        """
        Parameters:
            data: torch.Tensor [T, D] - 全銘柄を含む時系列データ
            label_columns: list[int] or list[str] - 予測対象（出力）としたいカラム
            feature_names: list[str] - データのカラム名（例: ['AAPL', 'GOOG', ...]）
        """
        self.data = data
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.stride = stride
        self.feature_names = feature_names

        D = data.shape[1]
        if label_columns is not None:
            if isinstance(label_columns[0], str):
                if feature_names is None:
                    raise ValueError("feature_names must be provided when label_columns are names.")
                self.label_indices = [feature_names.index(name) for name in label_columns]
            else:
                self.label_indices = label_columns
        else:
            self.label_indices = list(range(D))  # 全て対象

        self.input_indices = []
        self.label_indices_range = []

        total_window_size = input_width + shift + label_width - 1
        for start in range(0, data.shape[0] - total_window_size + 1, stride):
            end_input = start + input_width
            start_label = end_input + shift - 1
            end_label = start_label + label_width
            self.input_indices.append((start, end_input))
            self.label_indices_range.append((start_label, end_label))

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, idx):
        start_x, end_x = self.input_indices[idx]
        start_y, end_y = self.label_indices_range[idx]

        x = self.data[start_x:end_x]
        y = self.data[start_y:end_y][:, self.label_indices]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


    def to_dataloader(self, batch_size=32, shuffle=True, drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def sample_batch(self, batch_size=1, device='cpu'):
        x, y = next(iter(self.to_dataloader(batch_size=batch_size, shuffle=False)))
        return x.to(device), y.to(device)

    def plot(self, batch=None, feature=0, feature_name=None):
        if batch is None:
            batch = self.sample_batch()
        x, y = batch
        x = x[0].cpu().numpy()
        y = y[0].cpu().numpy()
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(x)), x[:, feature], label='Input', marker='.')
        plt.plot(np.arange(len(x), len(x) + len(y)), y[:, 0], label='Label', marker='s')
        plt.title(f"Feature: {feature_name or str(feature)}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
