# ========= window_dataset.py =========
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    """
    時系列データからウィンドウを作成し、入力とラベル系列を返すPyTorch Dataset。
    幾何リターンへの変換は行いません。
    """
    def __init__(self, df: pd.DataFrame, input_width: int, label_width: int, shift: int,
                 stride: int = 1, label_columns: list = None):

        # 1. パラメータ設定とインデックスの計算
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.stride = stride
        self.label_columns = label_columns
        self.total_window_size = input_width + shift

        self.column_indices = {name: i for i, name in enumerate(df.columns)}

        # 入力スライスとラベルスライス
        self.input_slice = slice(0, input_width)
        self.labels_slice = slice(self.total_window_size - label_width, self.total_window_size)

        # ラベル列のインデックス（データ全体に対する絶対インデックス）
        if label_columns is not None:
            self.label_indices_abs = [self.column_indices[name] for name in label_columns]
        else:
            # ラベル列指定がない場合は、すべての特徴量をラベルとして使用
            self.label_indices_abs = list(self.column_indices.values())

        # 2. ウィンドウの開始インデックスを計算
        max_start_index = len(self.data) - self.total_window_size
        self.start_indices = torch.arange(0, max_start_index + 1, stride)

    def __len__(self):
        """データセットに含まれるウィンドウの総数。"""
        return len(self.start_indices)

    def __getitem__(self, idx):
        """
        指定されたインデックスのウィンドウから入力とラベル系列を返す。
        """
        # ウィンドウの開始位置と終了位置
        start = self.start_indices[idx].item()
        end = start + self.total_window_size

        # ウィンドウ全体を抽出
        features = self.data[start:end, :]

        # 3. 入力とラベル系列の分割
        inputs = features[self.input_slice, :]
        labels_series = features[self.labels_slice, :]

        # 4. ラベル列の選択 (幾何リターン計算は削除)
        # shape: [label_width, num_labels]
        labels = labels_series[:, self.label_indices_abs]

        # inputs shape: [input_width, num_features]
        # labels shape: [label_width, num_labels] ← 累積リターンではない、系列データ
        return inputs, labels

class DataWindowPyTorch():
    """
    訓練・検証・テストのデータセットを管理し、PyTorch DataLoaderを提供するクラス。
    """
    def __init__(self, input_width: int, label_width: int, shift: int,
                 train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 stride: int = 1, shuffle: bool = True, label_columns: list = None,
                 batch_size: int = 32):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.stride = stride
        self.shuffle = shuffle
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.total_window_size = input_width + shift

        # プロット用のインデックス
        self.input_indices = np.arange(input_width)
        # ラベルのインデックス: ラベル系列の開始から終了まで
        self.label_indices = np.arange(self.total_window_size)[self.total_window_size - label_width:self.total_window_size]

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        if label_columns is not None:
            # ラベルテンソル内の列インデックス
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        else:
             # ラベル列指定がない場合、全列がラベルテンソル内のインデックスに対応
             self.label_columns_indices = self.column_indices

    def make_dataloader(self, df: pd.DataFrame, shuffle: bool = None) -> DataLoader:
        """DataFrameからTimeSeriesDatasetを作成し、DataLoaderを返します。"""
        if shuffle is None:
            shuffle = self.shuffle

        dataset = TimeSeriesDataset(
            df=df,
            input_width=self.input_width,
            label_width=self.label_width,
            shift=self.shift,
            stride=self.stride,
            label_columns=self.label_columns
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False
        )

    @property
    def train(self):
        """訓練用DataLoader"""
        return self.make_dataloader(self.train_df, shuffle=self.shuffle)

    @property
    def val(self):
        """検証用DataLoader"""
        return self.make_dataloader(self.val_df, shuffle=False)

    @property
    def test(self):
        """テスト用DataLoader"""
        return self.make_dataloader(self.test_df, shuffle=False)

    @property
    def sample_batch(self):
        """訓練データローダーから最初のバッチを取得し、キャッシュする"""
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        # PyTorch TensorをNumPyに変換してプロット用に準備
        inputs, labels = result
        return inputs.numpy(), labels.numpy()

    def plot(self, plot_col: str = 'y', max_subplots: int = 3):
        """ウィンドウの入力とラベル（系列）をプロットする"""
        inputs, labels = self.sample_batch

        plt.figure(figsize=(9, 6))
        plot_col_index = self.column_indices[plot_col] # 全特徴量に対するインデックス
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')

            # 入力のプロット（折れ線）
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

            # ラベル列インデックス取得 (ラベルテンソル内のインデックス)
            label_col_index = self.label_columns_indices.get(plot_col, None)

            if label_col_index is None:
                continue

            # ラベルの shape: (batch_size, label_width, num_labels)
            label_y = labels[n, :, label_col_index] # ラベル系列全体を取得

            # ラベルが系列（通常）
            plt.scatter(self.label_indices, label_y,
                    edgecolors='k', marker='s', label='Labels (Sequence)', c='green', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()
