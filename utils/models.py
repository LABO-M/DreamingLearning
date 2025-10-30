# utils/models.py

import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple

class LSTMWindowModel(nn.Module):
    """
    LSTM-based forecaster for windowed time series.
    Maps (B, input_width, num_features) ->
      - classification: (B, label_width, num_labels, num_classes)  [logits]
      - regression:     (B, label_width, num_labels)               [predictions]
    """
    def __init__(
        self,
        input_size: int,
        label_width: int,
        num_labels: int,
        task_type: Literal["classification", "regression"],
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        # classification only
        num_classes: Optional[int] = None,
        # optional MLP head depth
        mlp_hidden: Optional[int] = None,
    ):
        super().__init__()
        assert task_type in ("classification", "regression")
        self.task_type = task_type
        self.label_width = int(label_width)
        self.num_labels = int(num_labels)
        self.num_classes = int(num_classes) if num_classes is not None else None

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        enc_out_dim = hidden_size * (2 if bidirectional else 1)

        # Head: project encoder summary to the entire prediction horizon at once
        if task_type == "classification":
            if self.num_classes is None:
                raise ValueError("num_classes is required for classification.")
            out_dim = self.label_width * self.num_labels * self.num_classes
        else:
            out_dim = self.label_width * self.num_labels

        layers = []
        if mlp_hidden is not None and mlp_hidden > 0:
            layers += [nn.Linear(enc_out_dim, mlp_hidden), nn.ReLU(inplace=True)]
            layers += [nn.Linear(mlp_hidden, out_dim)]
        else:
            layers += [nn.Linear(enc_out_dim, out_dim)]
        self.head = nn.Sequential(*layers)

        # Optional layer norm on encoder summary for stability
        self.norm = nn.LayerNorm(enc_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_width, num_features)
        returns:
          classification: (B, label_width, num_labels, num_classes) logits
          regression:     (B, label_width, num_labels) predictions
        """
        # Encode the input window
        _, (h_n, _) = self.encoder(x)  # h_n: (num_layers * num_directions, B, hidden)
        # Take last layer's hidden states (and concat directions if biLSTM)
        if self.bidirectional:
            h_last_f = h_n[-2]  # (B, H)
            h_last_b = h_n[-1]  # (B, H)
            h = torch.cat([h_last_f, h_last_b], dim=1)
        else:
            h = h_n[-1]         # (B, H)

        h = self.norm(h)
        y = self.head(h)        # (B, out_dim)

        B = x.size(0)
        if self.task_type == "classification":
            y = y.view(B, self.label_width, self.num_labels, self.num_classes)
        else:
            y = y.view(B, self.label_width, self.num_labels)
        return y


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

class LSTMStudentT(nn.Module):
    """
    出力: [B, L, 3] = (mu, log_scale, log_nu)
      - mu        : 予測平均
      - log_scale : log σ（スケール, σ>0）
      - log_nu    : log ν（自由度, ν>2）
    既存の LSTMGaussian と同じ使い勝手で forward(x)->(out, hidden)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 proj_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_layernorm: bool = True):
        super().__init__()
        self.use_proj = proj_dim is not None and proj_dim != input_dim
        d_in = proj_dim or input_dim

        if self.use_proj:
            self.in_proj = nn.Linear(input_dim, proj_dim)
            self.ln_in = nn.LayerNorm(proj_dim) if use_layernorm else nn.Identity()
        else:
            self.in_proj = nn.Identity()
            self.ln_in = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()

        self.rnn = nn.LSTM(d_in, hidden_size, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.ln_h = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

        H = hidden_size
        self.head = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, 3)  # (mu, log_s, log_nu)
        )

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # x: [B, L, D]
        x = self.ln_in(self.in_proj(x))
        h, hidden = self.rnn(x, hidden)
        h = self.ln_h(h)
        out = self.head(h)  # [B, L, 3]
        return out, hidden

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
