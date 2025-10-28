# ========= window_dataset.py =========
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """
    data_list: List[Tensor[T, D]]  例) 銘柄ごとの系列テンソル
    input_window: int  (例: 60)
    label_window: int  (例: 20)  -> 予測/評価する長さ
    stride: int        (例: 1)   -> ウインドウ開始位置の間隔
    shuffle: bool      -> __getitem__順序ではなく、"インデックスリスト"をシャッフル
    drop_last: bool    -> 端数を捨てるか（DataLoader側でも指定可）
    """
    def __init__(self, data_list, input_window, label_window, stride=1, shuffle=True, drop_last=True):
        self.data_list = data_list
        self.I = int(input_window)
        self.L = int(label_window)
        self.stride = int(stride)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self._build_index()
        if self.shuffle:
            self._reshuffle()

    def _build_index(self):
        self.index = []
        for sid, x in enumerate(self.data_list):
            T, D = x.shape
            max_start = T - (self.I + self.L)
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                self.index.append((sid, start))
        self.N = len(self.index)

    def _reshuffle(self):
        import random
        random.shuffle(self.index)

    def on_epoch_start(self):
        if self.shuffle:
            self._reshuffle()

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        sid, start = self.index[i]
        x = self.data_list[sid]
        # inputs: [I, D], targets: [L, 1] (列0=ターゲット)
        inputs  = x[start : start + self.I, :]
        targets = x[start + self.I : start + self.I + self.L, [0]]
        return inputs, targets
