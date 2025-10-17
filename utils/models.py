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
class LSTMGaussian(nn.Module):
    """
    入力: x [B, L, D]  (D=1+n_exo, 列0=ターゲット)
    出力: out [B, L, 2]  ([...,0]=μ, [...,1]=logσ²), hidden_next
    """
    def __init__(self, input_dim, hidden_size=128, num_layers=1,
                 proj_dim=None, dropout=0.0, use_layernorm=False):
        super().__init__()
        self.use_proj = proj_dim is not None and proj_dim != input_dim
        self.in_dim = input_dim if not self.use_proj else proj_dim

        if self.use_proj:
            self.in_proj = nn.Linear(input_dim, proj_dim)
            self.ln_in = nn.LayerNorm(proj_dim) if use_layernorm else nn.Identity()

        self.rnn = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ヘッド：μ / logσ²
        self.head_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.head_lv = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.ln_head = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

    def forward(self, x, hidden=None):
        # x: [B,L,D]
        if self.use_proj:
            x = self.ln_in(self.in_proj(x))
        h, hidden_next = self.rnn(x, hidden)          # [B,L,H]
        h = self.ln_head(h)
        mu = self.head_mu(h)                          # [B,L,1]
        logvar = self.head_lv(h).clamp(-20, 10)       # 数値安定（Trainer側でも再clamp）
        out = torch.cat([mu, logvar], dim=-1)         # [B,L,2]
        return out, hidden_next




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
