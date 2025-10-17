# utils/models.py

import torch
import torch.nn as nn

# ----------------------------------
# LSTM モデル
# ----------------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.linear(out)
        return logits, hidden

# ----------------------------------
# GRU モデル（オプション）
# ----------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.gru(x, hidden)
        logits = self.linear(out)
        return logits, hidden

# ----------------------------------
# Transformer モデル（簡易版）
# ----------------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x) * (x.size(1) ** 0.5)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        logits = self.linear(x)
        return logits

# 補助：位置エンコーディング
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ----------------------------------
# LSTM モデル（価格予測用）
# ----------------------------------

class LSTMPriceModel(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 予測の平均と対数分散を出力するように変更
        self.fc = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out#, hidden



class PortfolioLSTMModel(nn.Module):
    def __init__(self, n_assets, hidden_dim=128, num_layers=2):
        super().__init__()
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(n_assets, hidden_dim, num_layers, batch_first=True)
        # ボラティリティ（各資産の分散）を予測
        self.vol_head = nn.Linear(hidden_dim, n_assets)
        # 相関（各ペアの相関係数）を予測
        n_correlations = n_assets * (n_assets - 1) // 2
        self.corr_head = nn.Linear(hidden_dim, n_correlations)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :] # 最後の時間ステップの隠れ状態
        vol_pred = self.vol_head(last_hidden_state)
        corr_pred = self.corr_head(last_hidden_state)
        return torch.exp(vol_pred), torch.tanh(corr_pred) # 分散は正、相関は-1~1に制約
